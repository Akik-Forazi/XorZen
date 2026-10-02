# XorZen Data Pipeline Documentation

The `data` sub-library provides a comprehensive ecosystem for cleaning, converting, loading, and augmenting datasets. It is optimized for high-throughput training using memory-mapped binary formats.

---

## 1. High-Level Utilities (`__init__.py`)
The `data` module exposes a streamlined API for common tasks:
- **Conversion**: `txt_to_bin`, `json_to_bin`, etc.
- **Loading**: `load_from_bin`, `load_from_dir`, `BinaryDataset`.
- **Validation**: `validate_data`.

---

## 2. File-by-File Breakdown

### `loader.py`
The primary entry point for data loading.
- **`BinaryDataset`**: A high-performance class for reading `.bin` files using `numpy.memmap`. It supports automatic metadata detection and sequence length validation.
- **`DirectoryLoader`**: Recursively scans directories and loads all supported files using the appropriate format handlers.
- **`load_from_*`**: Helper functions for specific formats (Parquet, JSON, TXT).

### `sharded_dataset.py` & `bin_dataset.py`
Optimized for multi-GPU or large-scale training.
- **`ShardedDataset`**: Handles datasets split across multiple `.npy` files. Uses binary search to map global indices to specific shards.
- **`BinDataset`**: Similar to `ShardedDataset` but specifically for raw `.bin` files, minimizing overhead.

### `converter.py`
Handles the transformation of raw text/json into tokenized binary formats.
- **`DataConverter`**: The core engine that orchestrates tokenization and binary packing.
- Supports multi-threaded conversion and progress tracking.

### `cleaner.py` & `validation.py`
Ensures data quality before training.
- **`DataCleaner`**: Removes boilerplate, handles encoding issues, and filters out low-quality text.
- **`DataValidator`**: Checks for corrupted files, sequence length consistency, and vocabulary overlap.

### `augmentation.py`
Implements on-the-fly data augmentation techniques:
- Synonym replacement.
- Back-translation.
- Masking/Noising for robustness.

### `inspector.py`
A diagnostic tool to "peek" into tokenized datasets.
- Decodes binary chunks back to text.
- Reports statistics on token distribution and sequence lengths.

### `sampling.py`
Implements various sampling strategies for training:
- **Weighted Sampling**: Balance different data sources.
- **Curriculum Sampling**: Start with easy examples and increase complexity.

### `formats/` (Sub-directory)
Modular handlers for different file types:
- `base.py`: The abstract base class for all format handlers.
- `txt.py`, `json.py`, `jsonl.py`, `gutenberg.py`: Format-specific logic for parsing raw data.

---

## 3. The Binary Format (.bin)
XorZen uses a custom binary format for training data to maximize I/O performance:
- **Header**: Stores metadata (version, dtype, sequence count).
- **Payload**: Raw token IDs packed as `uint16` or `uint32`.
- **Memory Mapping**: Allows training on datasets larger than available RAM by only loading the current batch into memory.

---
*Back to [README.md](../README.md)*
