# XorZen Tokenization Documentation

The `tokenizer` sub-library provides a production-grade, multi-format tokenization system designed for AGI-level tasks. It supports various algorithms, streaming architectures, and advanced vocabulary analysis.

---

## 1. Core API

### `loader.py`
The primary interface for loading tokenizers.
- **`load_pretrained(name)`**: Loads a standard XorZen tokenizer (e.g., `xorzen_32k`, `xorzen_65k`).
- **`XORZENXTokenizer`**: The main class for encoding text into token IDs and decoding them back.
- **Automatic Pathing**: Resolves tokenizer names to local JSON files in the `pretrained/` directory.

### `base.py`
Defines the architectural foundation.
- **`BaseTokenizer`**: Abstract class for all tokenizer implementations.
- **`TokenizerMetadata`**: Stores vocabulary size, special tokens, and algorithm details.
- **`TokenizerRegistry`**: Manages the mapping between names and physical files.

---

## 2. Advanced Features

### `advanced.py`
Extends basic tokenization with high-level configurations.
- **Algorithms**: Supports Byte-Level BPE, WordPiece, Unigram, and SentencePiece.
- **Normalization**: Implements NFD, Lowercase, and StripAccents pipelines.
- **Pre-tokenization**: Handles Whitespace, Digits, and Punctuation splitting.

### `streaming.py`
Optimized for massive datasets that don't fit in RAM.
- **Parallel Processing**: Uses `ThreadPoolExecutor` and `ProcessPoolExecutor` for multi-core encoding.
- **`TokenizationChunk`**: Emits tokenized data in manageable chunks.
- **`StreamingStats`**: Tracks throughput (tokens/sec) and progress in real-time.

### `special_tokens.py`
Manages the "thought" and "action" tokens critical for XorZen-Zero.
- Handles `<|user|>`, `<|assistant|>`, `<|think|>`, `<|reflect|>`, `<|call|>`, etc.
- Ensures these tokens are never split and have reserved IDs.

---

## 3. Training & Evaluation

### `trainer.py`
The engine for building new vocabularies from raw text.
- Supports training from files or memory-based iterators.
- Allows defining target vocabulary size and min-frequency constraints.

### `analysis.py` & `evaluation.py`
Diagnostic tools for vocabulary quality.
- **`TokenizerAnalyzer`**: Reports token frequency distributions and vocabulary coverage.
- **`TokenizerEvaluator`**: Measures compression ratios and fertility across different languages.

### `cache.py`
Implements a fast LRU cache for frequently used string-to-token mappings, reducing the overhead of repetitive calls during data conversion.

---

## 4. Ecosystem Integration

### `adapter.py`
Provides `xorzenTokenizerAdapter` to ensure compatibility between XorZen tokenizers and the HuggingFace `transformers` ecosystem, allowing XorZen models to use HF-style pipelines if needed.

---
*Back to [README.md](../README.md)*
