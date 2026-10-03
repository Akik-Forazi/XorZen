# Security Review — Phase: C++ Component Parity + Port Decision

**Scope**: Verify the existing encryption migration document. Do not modify the encryption system. Document any concrete safe improvements separately.

**Status**: NO CHANGES REQUIRED in this phase. The existing `docs/ENCRYPTION_MIGRATION.md` (337 lines) is comprehensive and accurate.

---

## What was verified

1. **Existing migration doc**: `docs/ENCRYPTION_MIGRATION.md` covers:
   - Where the master secret is defined (3 key parts at `xorm_crypto.py:120-122`)
   - How the secret is used (HKDF-SHA256 → AES-256-GCM key)
   - Why the current layout is "light obfuscation only, not real security"
   - A safe migration path (env var → key file → KMS)
   - Backward compatibility plan (read old format, write new format)

2. **No modifications made** to:
   - `xorzen/inference/xorm_crypto.py` (the secret constants)
   - `xorzen/inference/xorm_format.py` (the .xorm v2 format)
   - `xorzen/inference/xorm_runtime.py` (the runtime loader)
   - Any existing `.xorm` files

3. **No rotation performed** of the master secret.

4. **No master secret printed** in this report or in any of the parity docs.

5. **No `.xorm` files broken** — the encryption system was not touched.

---

## Concrete safe improvements (documented separately, NOT implemented)

These are independent of the C++ parity work and should be considered for a future phase:

### Improvement 1: Move the master secret to an environment variable

**Current state**: 24-byte master secret is split across three constants in `xorm_crypto.py:120-122`. This is documented as "light obfuscation only, not real security" (line 117-118).

**Safe improvement**: Read the master secret from `XORZEN_XORM_MASTER_SECRET` environment variable. Fall back to the hardcoded constants ONLY if the env var is not set (for backward compatibility). This is a 5-line change.

**Risk**: LOW — backward compatible (existing `.xorm` files still load via the fallback). The env var is optional.

**Implementation effort**: ~30 minutes (code change + tests).

### Improvement 2: Add a `xorm_rotate_secret.py` utility

**Current state**: No way to rotate the master secret without breaking existing `.xorm` files.

**Safe improvement**: Add a utility that:
1. Reads an old `.xorm` file with the current secret
2. Writes a new `.xorm` file with a user-supplied new secret
3. Verifies the new file can be loaded with the new secret

**Risk**: LOW — read-only on the old file, write-only on the new file. No mutation of existing files.

**Implementation effort**: ~1 hour (utility + tests).

### Improvement 3: Add HMAC-SHA256 tag verification to the .xorm v2 format

**Current state**: The .xorm v2 format already includes an HMAC-SHA256 tag (per `xorm_crypto.py`). But the verification is best-effort — if the tag is missing or wrong, the loader raises an error but does not log the failure.

**Safe improvement**: Add structured logging for HMAC verification failures (including the file path, expected vs actual tag prefix, and timestamp). This helps detect tampering attempts.

**Risk**: LOW — only adds logging, does not change behavior.

**Implementation effort**: ~30 minutes.

### Improvement 4: Document the threat model

**Current state**: The migration doc mentions "light obfuscation only, not real security" but does not articulate the threat model.

**Safe improvement**: Add a "Threat Model" section to `docs/ENCRYPTION_MIGRATION.md` that explicitly states:
- What the .xorm v2 encryption protects against (casual inspection, accidental disclosure)
- What it does NOT protect against (determined adversary with binary access, side-channel attacks)
- What the migration to env var / key file / KMS would protect against (binary-level extraction of the secret)

**Risk**: ZERO — documentation only.

**Implementation effort**: ~30 minutes.

---

## Why none of these are implemented in this phase

1. The user's directive was explicit: "Do not modify the encryption system in this phase unless necessary for correctness."
2. None of the improvements are required for correctness — they are security hardening.
3. The C++ parity work is the priority for this phase.
4. Mixing security changes with parity changes would conflate two independent concerns in the same commit history.

**Recommendation**: Implement Improvement 1 (env var fallback) and Improvement 4 (threat model doc) in a separate, dedicated security phase after the C++ parity work is complete. Improvements 2 and 3 can wait until there is a concrete need (e.g. an actual secret rotation event).
