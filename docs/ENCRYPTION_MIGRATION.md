# .xorm v2 Encryption Secret — Investigation & Migration Plan

**Scope:** `xorzen/inference/xorm_crypto.py`
**Audit date:** 2025
**Secret values are intentionally NOT reproduced in this document.** They are referenced only by variable name.

---

## 1. Where is the master secret defined?

| Item | Location | Notes |
|---|---|---|
| `_KEY_PART_A` | `xorzen/inference/xorm_crypto.py`, line **120** | 8-byte bytes literal |
| `_KEY_PART_B` | `xorzen/inference/xorm_crypto.py`, line **121** | 8-byte bytes literal |
| `_KEY_PART_C` | `xorzen/inference/xorm_crypto.py`, line **122** | 8-byte bytes literal |
| `_HKDF_SALT` | `xorzen/inference/xorm_crypto.py`, line **123** | `b"xorzen-xorm-v2-salt"` |
| `_HKDF_INFO` | `xorzen/inference/xorm_crypto.py`, line **124** | `b"xorzen-xorm-v2-aes-256-gcm"` |

The 24-byte master secret is the concatenation `_KEY_PART_A + _KEY_PART_B + _KEY_PART_C`,
assembled on lines **140** and **233**. The split-across-three-constants layout is **explicitly
documented in the file (lines 117–118) as light obfuscation only, not real security.**

A `grep` of the entire repository confirms the secret is referenced **only** inside
`xorzen/inference/xorm_crypto.py` (definition + HKDF calls) and re-exported indirectly
through `xorzen/inference/xorm_format.py` (which imports the helper functions, not the
constants themselves). No other file reads `_KEY_PART_*` or `_HKDF_*` directly.

---

## 2. How is the master secret used?

The same 24-byte master secret feeds **two independent HKDF-SHA256 derivations**:

### 2a. AES-256-GCM key (encryption / decryption)

Function: `_derive_aes_key(model_id)` — lines **127–147**

```
master = _KEY_PART_A + _KEY_PART_B + _KEY_PART_C        # 24 bytes
hkdf = HKDF(
    algorithm = SHA256,
    length    = 32,
    salt      = _HKDF_SALT,                              # b"xorzen-xorm-v2-salt"
    info      = _HKDF_INFO + b":" + model_id,            # b"xorzen-xorm-v2-aes-256-gcm:<model_id>"
)
aes_key = hkdf.derive(master)                            # 32 bytes
```

Per-model binding: `model_id` (taken from `manifest.json → model_name`) is mixed into the
HKDF `info` string so that two models sharing the same master secret still get distinct AES
keys. Extraction of one model's key therefore does **not** immediately reveal other models'
keys.

Consumers:
* `_encrypt_weights(state_dict_bytes, model_id)` — line 269
* `_decrypt_weights(ciphertext, nonce, tag, model_id)` — line 292

The AES-256-GCM nonce is 12 random bytes (`os.urandom(12)`) generated fresh on every
encrypt (line 271). The 16-byte GCM tag is split off from the ciphertext (lines 274–275).

### 2b. HMAC-SHA256 key (signature / tamper detection)

Function: `_compute_signature(...)` — lines **213–255**

```
master = _KEY_PART_A + _KEY_PART_B + _KEY_PART_C        # 24 bytes
hkdf = HKDF(
    algorithm = SHA256,
    length    = 32,
    salt      = _HKDF_SALT,                              # same salt as AES
    info      = b"xorzen-xorm-v2-hmac:" + model_id,     # DIFFERENT info string
)
hmac_key = hkdf.derive(master)                           # 32 bytes
```

Note the `info` strings differ: `xorzen-xorm-v2-aes-256-gcm:<model_id>` vs
`xorzen-xorm-v2-hmac:<model_id>`. This is the only thing preventing the AES key and HMAC
key from being identical (the salt is the same).

The HMAC covers a fixed concatenation of all critical .xorm v2 contents (lines 243–254):
`manifest || arch || weights.pt.enc || weights.pt.nonce || weights.pt.tag || license.json`,
each length-prefixed by a tag word (`b"manifest:"`, `b"\narch:"`, etc.). `hmac.compare_digest`
is used for constant-time comparison on verification (line 445).

---

## 3. Is the master secret part of the .xorm file format?

**No.** The master secret is **NOT stored inside the .xorm file** and is **NOT part of the
on-disk v2 format**.

What *is* stored in a v2 .xorm (per `xorm_format.py` lines 15–27 and `xorm_crypto.py`
lines 40–52):

| Stored artifact | Purpose |
|---|---|
| `.xorm_magic` (`"XORMV002\n"`) | Version sentinel |
| `manifest.json` | Model identity, `encrypted: true`, license metadata |
| `arch.json` | Architecture hyperparameters |
| `features.json` | Feature schema |
| `weights_manifest.json` | Layer shapes/dtypes (JSON, for tooling) |
| `weights.pt.enc` | AES-256-GCM ciphertext of `torch.save(state_dict)` |
| `weights.pt.nonce` | 12-byte GCM nonce |
| `weights.pt.tag` | 16-byte GCM authentication tag |
| `license.json` | Owner, allowed_uses, expiry, model_id |
| `signature.bin` | HMAC-SHA256 over the above |
| `meta.json` | Training metrics |

The master secret (and its derived AES/HMAC keys) lives **only** in the runtime source code
(`xorm_crypto.py`). The .xorm file itself is "key-less" — it stores everything needed to
verify and decrypt *given the right runtime secret*, but nothing that reveals the secret.

**Consequence for v2 files:** Because the secret is not in the file, a v2 .xorm is
self-consistent only with respect to *one specific master secret*. Change the secret in
the runtime and every existing v2 file's signature check (`verify_xorm_signature`) and
weight decryption (`_decrypt_weights`) will fail. See §5–6.

---

## 4. Can the master secret be migrated to an environment variable?

**Yes — and it is straightforward.** The secret is read from module-level constants in
exactly two places (lines 140 and 233). Both assemble the master secret the same way:

```python
master = _KEY_PART_A + _KEY_PART_B + _KEY_PART_C
```

A migration would replace these two reads with a single resolver function that prefers an
environment variable, falling back to the compiled-in constants for backward compatibility.

### Proposed resolver

```python
import os

_XORM_SECRET_ENV_VAR = "XORZEN_XORM_MASTER_SECRET"

def _resolve_master_secret() -> bytes:
    """
    Resolve the .xorm v2 master secret.

    Priority:
      1. XORZEN_XORM_MASTER_SECRET environment variable (hex or base64).
      2. Compiled-in fallback: _KEY_PART_A + _KEY_PART_B + _KEY_PART_C.

    The fallback MUST stay equal to the historical compiled-in value, or
    every existing v2 .xorm file will fail signature/decryption at load time.
    """
    env_val = os.environ.get(_XORM_SECRET_ENV_VAR)
    if env_val:
        # Accept hex (48 hex chars → 24 bytes) or base64 (32 b64 chars → 24 bytes).
        try:
            if len(env_val) == 48:
                return bytes.fromhex(env_val)
            import base64
            return base64.b64decode(env_val)
        except ValueError as e:
            raise ValueError(
                f"{_XORM_SECRET_ENV_VAR} is set but could not be decoded as hex or base64: {e}"
            )
    # Backward-compat fallback — keep the literal concatenation exactly as it was.
    return _KEY_PART_A + _KEY_PART_B + _KEY_PART_C
```

Then replace the two `master = _KEY_PART_A + _KEY_PART_B + _KEY_PART_C` lines (140 and 233)
with `master = _resolve_master_secret()`.

### Why hex/base64 and not raw bytes?

Environment variables are strings on every OS. The secret is 24 raw bytes; hex (48 chars)
or standard base64 (32 chars) is the standard encoding. Hex is preferable because it is
case-insensitive and unambiguous.

### Optional: length validation

`_resolve_master_secret()` can additionally assert `len(master) >= 16` (HKDF accepts any
IKM length, but the historical value is 24 bytes — enforcing a minimum keeps the security
claim of ≥128-bit entropy).

### Operational requirements

* Whoever distributes v2 .xorm files must set `XORZEN_XORM_MASTER_SECRET` to the *same*
  value used at encrypt time, on every machine that needs to read those files.
* The env var must be set before `import xorzen.inference.xorm_crypto` is *first* triggered
  through a v2 read/write. The resolver is called inside `_derive_aes_key` / 
  `_compute_signature` (both at call time, not import time), so this is per-operation —
  setting the env var mid-process is safe and takes effect on the next v2 read/write.
* For deployments where the secret cannot be in process env (e.g. container images), the
  same pattern extends naturally to: a file path env var (`XORZEN_XORM_SECRET_FILE`), a
  keyring/secret-manager lookup, or an HSM-backed KMS call. The resolver is the single
  chokepoint.

---

## 5. Backward-compatibility risks

### Risk 1 — **All existing v2 .xorm files become unreadable if the secret changes.**
This is the dominant risk. The AES key and HMAC key are deterministic functions of
`(master_secret, salt, info, model_id)`. Changing any input changes both keys, which:
* breaks `verify_xorm_signature` (HMAC mismatch → "file has been tampered with"),
* breaks `_decrypt_weights` (AES-GCM tag verification failure → ValueError raised at line
  300-301).

**Impact:** silent-looking "tampering detected" errors on every pre-existing v2 file,
with no in-band hint that the cause is a key rotation rather than actual tampering.

### Risk 2 — **`encrypt_existing_xorm` cannot be re-run without the original secret.**
`encrypt_existing_xorm` (lines 317–407) converts a v1 .xorm to v2 by reading the plaintext
weights and re-encrypting. If the secret has changed, re-running this on an *already
encrypted v2* file is impossible — the function explicitly rejects v2 inputs (line 339–340).
The only path back to "freshly encrypted with the new secret" is to start from a v1 .xorm
(or from the live `torch.nn.Module` in memory).

### Risk 3 — **No version field inside `signature.bin` or `weights.pt.enc`.**
The v2 format has no key-version header. There is no way for the runtime to know *which*
secret was used to produce a given v2 file, so it cannot try multiple secrets or fall back
gracefully. A v2 file either matches the current secret or it doesn't.

### Risk 4 — **Test coverage of v2 is effectively zero.**
`tests/test_v101_regression.py` exercises the .xorm round-trip but only via
`XormWriter(..., encrypt=False)` (v1 path). There is no test that creates a v2 file with
one secret and reads it back with the same secret, let alone a rotation test. Any
migration will be flying blind unless tests are added first.

### Risk 5 — **`model_id` participates in key derivation.**
The `model_id` fed into HKDF comes from `manifest.json → model_name` (read at lines 352,
433, 492). Renaming a model after encryption (without re-encrypting) would break loading
in exactly the same way a secret change would. This is orthogonal to the migration but
worth noting: any tooling that lets users rename models must re-encrypt.

### Risk 6 — **Environment variable availability across processes.**
If the env var is set in a parent process but not propagated (e.g. systemd unit missing
`Environment=`, Docker container missing `-e`, Jupyter kernel started before the var was
exported), the resolver silently falls back to the compiled-in secret. A v2 file encrypted
with the env-var secret will then fail to load with the same misleading "tampering
detected" error.

### Risk 7 — **The compiled-in fallback remains a leak vector.**
Even after migration, anyone with read access to `xorm_crypto.py` (e.g. anyone who can
`pip download xorzen`) can extract the fallback secret and decrypt any v2 file that was
encrypted with it. The migration improves operational key management but does not, by
itself, improve confidentiality against source-code-level attackers. Real remediation
requires (a) encrypting all future v2 files with an env-var-only secret AND (b) eventually
removing the fallback constant from the source (which then permanently orphans any v2 file
not yet re-encrypted).

### Risk 8 — **Distributed training / multi-process loaders.**
If v2 files are loaded by DataLoader workers or DDP ranks, every worker process must see
the same `XORZEN_XORM_MASTER_SECRET`. This is usually automatic via env inheritance, but
torch multiprocessing spawn strategies (`spawn` vs `fork`) and Ray/Dask workers may need
explicit propagation.

---

## 6. Can existing v2 .xorm files remain readable if the secret changes?

**Not without explicit re-encryption.** There is no in-place path. Concretely:

| Scenario | Existing v2 files readable? | Action required |
|---|---|---|
| Secret unchanged (no migration) | ✅ Yes | None |
| Secret rotated, env-var migration with **same fallback value** | ✅ Yes — *as long as* the env var is unset on the loading machine (falls back to original). ⚠️ Fragile: a single machine with the env var set sees a different secret and breaks. | None for read; recommend explicitly documenting which secret was used per file. |
| Secret rotated, fallback constant also changed in source | ❌ No | Re-encrypt every v2 file from a v1 source (or from a live model) using the new secret. |
| Secret rotated, env var set on all machines, fallback kept as original | ✅ Yes — *for files encrypted under the env-var secret*. Files encrypted under the old compiled-in secret become unreadable on those machines. | Re-encrypt once per model. |
| Secret rotated, env var set on **some** machines only | ❌ Mixed | Files encrypted with env-var secret fail on machines without the var; files encrypted with compiled-in secret fail on machines with the var. Effectively two incompatible populations. |

### Re-encryption procedure (per model)

For each existing v2 .xorm file `model.xorm`:

1. **Find a v1 source.** If only v2 is available, you must first decrypt it with the
   *original* secret to recover the plaintext state_dict. (This is only possible on a
   machine that still has the original secret — env var or compiled-in.)
2. **Decrypt to a v1 file** (or to an in-memory `state_dict`). The codebase does not have
   a public "v2 → v1 downgrade" helper, so this is a small new utility:
   ```python
   sd = load_encrypted_state_dict("model.xorm", intended_use="inference")
   # then write a v1 .xorm via XormWriter(encrypt=False, ...)
   ```
3. **Re-encrypt with the new secret** using `encrypt_existing_xorm(v1_path, v2_path,
   license=...)`. Ensure `XORZEN_XORM_MASTER_SECRET` is set to the new value in the
   process that runs this.
4. **Verify** with `verify_xorm_signature(v2_path)` on a machine running the new secret.
5. **Archive or destroy** the intermediate v1 file (it contains plaintext weights).

### Recommended migration sequencing (lowest risk first)

1. **Phase 0 — Tests.** Add a v2 round-trip test (`encrypt=True`, write, read back,
   assert state_dict equal) using the current compiled-in secret. Also add a test that
   asserts `XORZEN_XORM_MASTER_SECRET` env var, when set, is honored by `_resolve_master_secret`.
2. **Phase 1 — Add the resolver, keep the fallback byte-for-byte identical.** This is a
   no-op for existing files. Ship it. Nothing breaks. Existing v2 files keep loading.
3. **Phase 2 — Operationally set the env var on the encrypting/build machine.** New v2
   files are now produced under the env-var secret. Existing v2 files still load on
   machines where the env var is *not* set (they fall back to compiled-in).
4. **Phase 3 — Re-encrypt existing v2 files.** For each model: decrypt with old secret,
   re-encrypt with new (env-var) secret, replace the file. Keep both secrets available
   during this transition window.
5. **Phase 4 — Distribute the env var to all readers.** Once every consumer machine has
   `XORZEN_XORM_MASTER_SECRET` set, all reads use the new secret. Compiled-in fallback is
   now dead code for production but still keeps legacy/untouched files loadable on dev
   machines without the env var.
6. **Phase 5 — Optional: remove the compiled-in fallback.** Only after every v2 file in
   circulation has been re-encrypted under the env-var secret and every reader has the
   env var set. Removing the fallback orphans any v2 file that was not re-encrypted.

### Things to NOT do

* **Do not** change `_KEY_PART_A/B/C` or `_HKDF_SALT`/`_HKDF_INFO` as part of the env-var
  migration. The HKDF output is a function of all five inputs; changing any of them
  produces a different AES/HMAC key and breaks every existing v2 file. Key rotation
  (changing the secret itself) is a *separate* operation from env-var migration (changing
  *where* the secret is stored) and should be staged independently.
* **Do not** introduce a "try multiple secrets" fallback in `verify_xorm_signature` or
  `_decrypt_weights` to ease rotation — that silently masks tampering and defeats the
  HMAC's purpose. If multi-secret support is genuinely needed, add an explicit key-version
  field to the v2 format (e.g. `manifest.json → "key_version": 1`) and dispatch on it
  explicitly, with a hard failure if the version is unknown.
* **Do not** log the resolved secret value, even at DEBUG level, in any error path. The
  "tampering detected" error at line 300 must continue to report only the exception
  string from `AESGCM.decrypt`, never the key.
* **Do not** put the env-var secret in a Jupyter notebook cell or any committed
  `.env` file. Use the OS keychain, a secrets manager, or a CI/CD protected variable.

---

## 7. Summary

| Question | Answer |
|---|---|
| Where is the master secret defined? | `xorm_crypto.py` lines 120–124 (`_KEY_PART_A/B/C`, `_HKDF_SALT`, `_HKDF_INFO`). Concatenated to 24 bytes on lines 140 and 233. |
| How is it used? | HKDF-SHA256 twice: once (info=`...-aes-256-gcm:<model_id>`) → 32-byte AES-256-GCM key; once (info=`...-hmac:<model_id>`) → 32-byte HMAC-SHA256 key. Salt is identical for both. |
| Is it part of the .xorm file format? | No. The .xorm v2 format stores ciphertext, nonce, tag, license, and HMAC signature, but never the master secret or the derived keys. The secret lives only in the runtime source. |
| Can it be migrated to an env var? | Yes. Add a `_resolve_master_secret()` helper that prefers `XORZEN_XORM_MASTER_SECRET` (hex or base64) and falls back to the existing constant concatenation. Two-line change at the two `master = ...` sites. |
| Backward-compat risks? | (1) All existing v2 files break if the secret value changes. (2) No re-encrypt-in-place helper. (3) No key-version field in the format. (4) No v2 test coverage. (5) `model_id` participates in derivation. (6) Env var propagation hazards. (7) Compiled-in fallback stays a leak vector. (8) Multi-process propagation. |
| Can existing v2 files stay readable if the secret changes? | Not automatically. Each existing v2 file must be re-encrypted under the new secret, which requires access to the original secret for the decrypt step. Sequenced migration (Phase 0–5 above) keeps the system continuously readable if env-var migration is done *before* key rotation. |
