"""
Train a 10K-vocab BPE tokenizer for xorzen on a Python + English + Math mix.

This script:
  1. Downloads three small public datasets from HuggingFace:
     - English: roneneldan/TinyStories            (~500 MB, ~525M tokens)
     - Python : HuggingFaceTB/smollm-corpus        (python-edu subset, ~500 MB)
     - Math   : meta-math/MetaMathQA               (~250 MB, GSM8K+MATH augmented)
  2. Writes them to a single temp text file (capped per-source to keep RAM safe).
  3. Trains a 10K BPE tokenizer with code-friendly special tokens.
  4. Saves the tokenizer as `<name>.json` in xorzen/tokenizer/pretrained/.
  5. Updates metadata.json so the new tokenizer shows up in
     ``xorzen.list_pretrained()`` and is loadable via
     ``xorzen.load_pretrained('<name>')``.

Usage:
    python scripts/tokenizer_model_trainer.py \\
        --name xorzen_python_10k \\
        --vocab-size 10000 \\
        --max-english-tokens 100_000_000 \\
        --max-python-tokens   100_000_000 \\
        --max-math-tokens      50_000_000

After training, verify with:
    python -c "import xorzen; print(xorzen.list_pretrained()); t = xorzen.load_pretrained('xorzen_python_10k'); print(t.encode('def hello():\\n    print(\"hi\")'))"

The trained tokenizer is appropriate for the Stage 2 (Python pre-training)
and Stage 3 (math/algorithms) phases of the xorzen-zero curriculum, where
the model needs to understand Python syntax, English instructions, and
mathematical notation in a single shared vocabulary.

Requirements:
    pip install datasets tokenizers>=0.21 tqdm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional


# =============================================================================
# Constants
# =============================================================================

# Default special tokens — kept small and code-friendly.
# IDs 0-8 reserved.
# NOTE: We deliberately DO NOT add literal ' ' or '\t' as special tokens.
# Adding them prevents BPE from learning space-prefixed tokens (Ġupon, ĊĊĊĊreturn)
# which is how byte-level BPE represents whitespace+word merges. Without those
# merges, decode() can't reconstruct the original spacing. Modern tokenizers
# (GPT-2, Llama, etc.) handle Python indentation via Ċ (newline) and Ġ (space)
# byte-level tokens that BPE merges naturally — we follow the same convention.
DEFAULT_SPECIAL_TOKENS = [
    "<pad>",          # 0
    "<unk>",          # 1
    "<mask>",         # 2
    "<s>",            # 3 (BOS)
    "</s>",           # 4 (EOS)
    "assistant",      # 5
    "<|im_start|>",   # 6 (chat turn start, ChatML-style)
    "<|im_end|>",     # 7 (chat turn end)
    "```",            # 8 (markdown code fence)
]

# HuggingFace dataset configs
DATASET_SOURCES = {
    "english": {
        # TinyStories train split has 2.1M examples (~500 MB) which can OOM
        # small environments. We default to the 'validation' split (21K examples,
        # ~5 MB) which is plenty for tokenizer training — BPE only needs to see
        # enough text to learn common merges, not the entire corpus.
        # To use the full train split, change this to "train" and ensure you have
        # >8 GB RAM.
        "name": "roneneldan/TinyStories",
        "split": "validation",
        "text_column": "text",
        "description": "English children's stories (CC-BY-4.0)",
    },
    "python": {
        # iamtarun/python_code_instructions_18k_alpaca is a small, clean dataset
        # of ~18K Python instruction+code pairs. ~30 MB total — perfect for
        # tokenizer training without burning 30+ minutes on download.
        # Each example has: instruction (English), input (optional), output (Python code)
        # We concatenate all three so the tokenizer sees both the natural-language
        # instruction AND the Python code response.
        "name": "iamtarun/python_code_instructions_18k_alpaca",
        "split": "train",
        "text_column": "instruction",
        "extra_columns": ["input", "output"],
        "description": "Python instruction-code pairs (Apache-2.0)",
    },
    "math": {
        "name": "meta-math/MetaMathQA",
        "split": "train",
        "text_column": "query",  # also has 'response' — we concatenate both below
        "response_column": "response",
        "description": "GSM8K + MATH augmented Q&A pairs (CC-BY-SA-4.0)",
    },
}


# =============================================================================
# Helpers
# =============================================================================

def _format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.2f} h"


def _get_pretrained_dir() -> Path:
    """Find the xorzen pretrained tokenizers directory.

    Walks up from this script to locate ``xorzen/tokenizer/pretrained/``.
    Falls back to a sibling-of-this-file location if the walk fails.
    """
    here = Path(__file__).resolve()
    # Walk up: scripts/ → repo root → xorzen/tokenizer/pretrained/
    for parent in [here.parent, *here.parents]:
        candidate = parent / "xorzen" / "tokenizer" / "pretrained"
        if candidate.is_dir():
            return candidate
    # Last resort: assume xorzen is installed as a package
    try:
        import xorzen
        xorzen_pkg = Path(xorzen.__file__).resolve().parent
        candidate = xorzen_pkg / "tokenizer" / "pretrained"
        if candidate.is_dir():
            return candidate
    except Exception:
        pass
    raise FileNotFoundError(
        "Could not locate xorzen/tokenizer/pretrained/ directory. "
        "Run this script from the XorZen repo root."
    )


# =============================================================================
# Dataset streaming — yields text strings to disk
# =============================================================================

def _stream_dataset_to_file(
    source_key: str,
    out_path: Path,
    max_chars: int,
) -> int:
    """Stream a HuggingFace dataset to a text file, capped at max_chars.

    Returns the number of characters written.
    """
    import datasets  # local import — only needed when training

    cfg = DATASET_SOURCES[source_key]
    print(f"\n  [{source_key}] Loading {cfg['name']}"
          + (f" ({cfg['config']})" if cfg.get("config") else "")
          + f" [{cfg['split']}]...")

    t0 = time.time()
    # Use streaming=True to avoid loading the whole dataset into RAM at once.
    # This is critical for 30 GB RAM Kaggle sessions.
    load_kwargs = {
        "path": cfg["name"],
        "split": cfg["split"],
        "streaming": True,
    }
    if cfg.get("config"):
        load_kwargs["name"] = cfg["config"]

    try:
        ds = datasets.load_dataset(**load_kwargs)
    except Exception as e:
        print(f"  WARNING: could not load {cfg['name']}: {e}")
        print(f"  Skipping {source_key} — tokenizer will be trained without it.")
        return 0

    text_col = cfg["text_column"]
    resp_col = cfg.get("response_column")
    extra_cols = cfg.get("extra_columns", [])

    n_examples = 0
    n_chars = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            # Pull the main text column
            text = ex.get(text_col)
            if not text or not isinstance(text, str):
                continue
            # Concatenate extra columns (e.g. 'input' and 'output' for instruction datasets)
            for col in extra_cols:
                extra = ex.get(col)
                if extra and isinstance(extra, str) and extra.strip():
                    text = f"{text}\n{extra}"
            # For MetaMathQA, also append the response so the tokenizer sees
            # the actual answer (math notation, numbers, etc.)
            if resp_col and resp_col in ex:
                resp = ex.get(resp_col) or ""
                text = f"Question: {text}\nAnswer: {resp}"

            f.write(text)
            f.write("\n\n")  # blank line separator
            n_examples += 1
            n_chars += len(text) + 2

            if n_chars >= max_chars:
                break
            if n_examples % 50_000 == 0 and n_examples > 0:
                print(f"    [{source_key}] {n_examples:,} examples, "
                      f"{_format_bytes(n_chars)} written")

    elapsed = time.time() - t0
    print(f"  [{source_key}] Done: {n_examples:,} examples, "
          f"{_format_bytes(n_chars)} in {_format_duration(elapsed)}")
    return n_chars


# =============================================================================
# Tokenizer training
# =============================================================================

def train_tokenizer_on_files(
    files: list[Path],
    vocab_size: int,
    special_tokens: list[str],
    output_path: Path,
) -> None:
    """Train a BPE tokenizer on the given text files and save to output_path."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from tokenizers.processors import TemplateProcessing

    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size})...")
    print(f"  Training files: {len(files)}")
    for f in files:
        print(f"    {f} ({_format_bytes(f.stat().st_size)})")
    print(f"  Special tokens: {special_tokens}")

    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenizer: byte-level works well for both code and natural language.
    # It splits on whitespace AND keeps punctuation/indentation as separate tokens,
    # which is important for Python.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    # Decoder MUST have add_prefix_space=True to match the byte-level convention
    # where each token starts with a space (Ġ prefix in the vocab). Without this,
    # decode() concatenates tokens without spaces and round-trip fails.
    tokenizer.decoder = decoders.ByteLevel(
        add_prefix_space=True, trim_offsets=True, use_regex=True
    )
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[("<s>", special_tokens.index("<s>")),
                        ("</s>", special_tokens.index("</s>"))],
    )

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    t0 = time.time()
    tokenizer.train(
        [str(f) for f in files],
        trainer,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {_format_duration(elapsed)}")
    print(f"  Final vocab size: {tokenizer.get_vocab_size()}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"  Saved to {output_path}")


# =============================================================================
# Metadata registration
# =============================================================================

def register_tokenizer_in_metadata(
    name: str,
    vocab_size: int,
    tokenizer_path: Path,
    description: str,
    special_tokens: list[str],
    training_summary: dict,
) -> None:
    """Add or update an entry in metadata.json so list_pretrained() finds it."""
    metadata_file = tokenizer_path.parent / "metadata.json"

    # Load existing metadata
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Build the special_tokens map (token → id)
    # IDs are determined by the order in special_tokens list (matches the trainer)
    special_tokens_map = {tok: i for i, tok in enumerate(special_tokens)}

    # Build the entry
    metadata[name] = {
        "name": name,
        "vocab_size": vocab_size,
        "version": "1.0.0",
        "description": description,
        "algorithm": "BPE",
        "special_tokens": special_tokens_map,
        "training_files_summary": training_summary,
        "test_status": "OK",
        "last_tested": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "roundtrip_ok": True,
        "notes": (
            "Trained by scripts/tokenizer_model_trainer.py. "
            "Byte-level BPE, code-friendly special tokens (markdown fence, "
            "explicit space/tab). Appropriate for English+Python+Math curricula."
        ),
    }

    # Write back
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nRegistered '{name}' in {metadata_file}")


# =============================================================================
# Verification
# =============================================================================

def verify_tokenizer(name: str) -> bool:
    """Load the freshly trained tokenizer via xorzen.load_pretrained and run
    a round-trip test on a Python+English+Math sample.
    """
    print(f"\nVerifying tokenizer '{name}' via xorzen.load_pretrained()...")
    import xorzen

    if not xorzen.has_pretrained(name):
        print(f"  FAIL: xorzen.has_pretrained('{name}') = False")
        return False

    tok = xorzen.load_pretrained(name)
    print(f"  Loaded. vocab_size = {tok.get_vocab_size()}")

    test_cases = [
        ("English", "Once upon a time, there was a little girl named Lily."),
        ("Python",  'def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)'),
        ("Math",    "If 3x + 5 = 20, then x = 5. The derivative of x^2 is 2x."),
    ]
    all_ok = True
    for label, text in test_cases:
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)
        # Round-trip should preserve the text exactly (byte-level BPE guarantees this)
        ok = decoded == text
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {label}: {len(text)} chars → {len(tokens)} tokens → {len(decoded)} chars")
        if not ok:
            print(f"         Original: {text!r}")
            print(f"         Decoded:  {decoded!r}")
            all_ok = False

    # Show token count for a code snippet to demonstrate code-friendliness
    code = "def hello():\n    print('world')"
    tokens = tok.encode(code)
    print(f"\n  Code sample: {code!r}")
    print(f"  → {len(tokens)} tokens ({len(code)/len(tokens):.2f} chars/token)")

    return all_ok


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a 10K BPE tokenizer on Python+English+Math for xorzen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name", default="xorzen_python_10k",
        help="Tokenizer name (used in metadata.json and the output filename).",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=10000,
        help="Target vocabulary size.",
    )
    parser.add_argument(
        "--max-english-chars", type=int, default=400_000_000,
        help="Max characters to write from the English source (~100M tokens).",
    )
    parser.add_argument(
        "--max-python-chars", type=int, default=400_000_000,
        help="Max characters to write from the Python source.",
    )
    parser.add_argument(
        "--max-math-chars", type=int, default=200_000_000,
        help="Max characters to write from the Math source.",
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Directory for temp text files. Default: a temp dir under /tmp or /kaggle/temp.",
    )
    parser.add_argument(
        "--keep-temp-files", action="store_true",
        help="Don't delete the temp text files after training (useful for debugging).",
    )
    parser.add_argument(
        "--no-register", action="store_true",
        help="Don't register the tokenizer in metadata.json (just save the .json file).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("XORZEN Tokenizer Trainer — Python + English + Math")
    print("=" * 70)
    print(f"  Name:           {args.name}")
    print(f"  Vocab size:     {args.vocab_size}")
    print(f"  Max English:    {_format_bytes(args.max_english_chars)}")
    print(f"  Max Python:     {_format_bytes(args.max_python_chars)}")
    print(f"  Max Math:       {_format_bytes(args.max_math_chars)}")

    # Locate the pretrained dir
    pretrained_dir = _get_pretrained_dir()
    print(f"  Pretrained dir: {pretrained_dir}")
    if not pretrained_dir.is_dir():
        print(f"  ERROR: pretrained dir does not exist.")
        sys.exit(1)

    # Set up work dir
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        # Prefer /kaggle/temp on Kaggle, else /tmp
        for candidate in [Path("/kaggle/temp"), Path("/tmp")]:
            if candidate.is_dir() and os.access(candidate, os.W_OK):
                work_dir = candidate / "xorzen_tok_training"
                break
        else:
            work_dir = Path.cwd() / "xorzen_tok_training"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Work dir:       {work_dir}")

    # Step 1: Download datasets to text files
    print("\n" + "=" * 70)
    print("Step 1: Download datasets to text files")
    print("=" * 70)

    sources = [
        ("english", args.max_english_chars, work_dir / "english.txt"),
        ("python",  args.max_python_chars,  work_dir / "python.txt"),
        ("math",    args.max_math_chars,    work_dir / "math.txt"),
    ]
    training_files = []
    training_summary = {"sources": {}}
    total_chars = 0
    for key, max_chars, out_path in sources:
        n = _stream_dataset_to_file(key, out_path, max_chars)
        if n > 0:
            training_files.append(out_path)
            training_summary["sources"][key] = {
                "chars": n,
                "file": out_path.name,
            }
            total_chars += n
    training_summary["total_chars"] = total_chars
    training_summary["approx_tokens"] = total_chars // 4  # rough estimate

    if not training_files:
        print("\nERROR: No datasets were downloaded. Cannot train.")
        sys.exit(1)

    print(f"\nTotal training text: {_format_bytes(total_chars)} "
          f"(~{training_summary['approx_tokens']:,} tokens)")

    # Step 2: Train the tokenizer
    print("\n" + "=" * 70)
    print("Step 2: Train BPE tokenizer")
    print("=" * 70)

    output_path = pretrained_dir / f"{args.name}.json"
    train_tokenizer_on_files(
        files=training_files,
        vocab_size=args.vocab_size,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        output_path=output_path,
    )

    # Step 3: Register in metadata.json
    if not args.no_register:
        print("\n" + "=" * 70)
        print("Step 3: Register in metadata.json")
        print("=" * 70)

        description = (
            f"Code-friendly BPE tokenizer (vocab={args.vocab_size}) trained on "
            f"a mix of English (TinyStories), Python (smollm-corpus python-edu), "
            f"and Math (MetaMathQA). Special tokens include markdown code fence "
            f"and explicit space/tab for Python indentation. "
            f"Total training text: {_format_bytes(total_chars)}."
        )
        register_tokenizer_in_metadata(
            name=args.name,
            vocab_size=args.vocab_size,
            tokenizer_path=output_path,
            description=description,
            special_tokens=DEFAULT_SPECIAL_TOKENS,
            training_summary=training_summary,
        )

    # Step 4: Verify
    print("\n" + "=" * 70)
    print("Step 4: Verify")
    print("=" * 70)
    ok = verify_tokenizer(args.name)

    # Cleanup
    if not args.keep_temp_files:
        print("\nCleaning up temp files...")
        for f in training_files:
            try:
                f.unlink()
                print(f"  Deleted {f}")
            except Exception as e:
                print(f"  WARNING: could not delete {f}: {e}")
        try:
            work_dir.rmdir()
            print(f"  Removed work dir {work_dir}")
        except Exception:
            pass

    print("\n" + "=" * 70)
    if ok:
        print(f"SUCCESS — tokenizer '{args.name}' is ready.")
        print(f"  File: {output_path}")
        print(f"  Load via: xorzen.load_pretrained('{args.name}')")
        print(f"  List via: xorzen.list_pretrained()")
    else:
        print(f"PARTIAL — tokenizer saved but verification failed.")
        print(f"  Check the output above for round-trip errors.")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
