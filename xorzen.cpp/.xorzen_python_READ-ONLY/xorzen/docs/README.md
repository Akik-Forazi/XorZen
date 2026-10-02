# XorZen Documentation Index

Welcome to the official documentation for **XorZen 0.2.4**. This framework is a state-of-the-art deep learning environment optimized for training large-scale, efficient models on consumer hardware.

---

## 📚 Documentation Modules

Navigate the codebase with these detailed technical manuals:

### 1. [Core & Framework Foundation](core.md)
*`config.py`, `train_superintelligence.py`, `exceptions.py`*
Learn about the DNA of XorZen models, high-level AGI pipelines, and the robust error handling system.

### 2. [Architecture & Components](model.md)
*`hass_block.py`, `zmoe.py`, `ssm.py`, `cot_vector.py`*
Deep dive into the Hybrid Attention-Shard Switch, Disk-based MoE, and Latent Chain-of-Thought.

### 3. [Model Variants & Registry](models.md)
*`zero/`, `igris/`, `registry.py`*
Explore the Zero and Igris model families, scaling from 23K to 7B+ parameters.

### 4. [Training & Learning Strategies](training.md)
*`trainer.py`, `curriculum.py`, `checkpoint.py`*
Documentation for the standard trainer, curriculum-based learning, and versioned state persistence.

### 5. [Speed & CPU Optimization](speed.md)
*`booster.py`, `csrc/`, `setup_speed.py`*
How XorZen achieves GPU-equivalent speeds on CPU using C++ kernels and AVX2 optimizations.

### 6. [Data Pipeline & Tokenization](data.md) & [Tokenizer Details](tokenizer.md)
*`loader.py`, `converter.py`, `sharded_dataset.py`, `advanced.py`*
Efficient binary data conversion, streaming tokenization, and AGI-specific vocabulary management.

### 7. [Utilities, Transfer & GUI](utils_transfer.md)
*`math_utils.py`, `sharding.py`, `gguf_transfer.py`, `dashboard.py`*
The mathematical engine, model transfer tools, and real-time monitoring interface.

---

## 🚀 Getting Started

1.  **Configure**: Set your model parameters in `config.py`.
2.  **Optimize**: Run `python speed/setup_speed.py build_ext --inplace` to compile the lightning kernels.
3.  **Data**: Convert your text data using `xorzen.data.txt_to_bin`.
4.  **Train**: Use `train_superintelligence.py` or the standard `XORZENXTrainer` to begin.

---
*Created by Akik Faraji, Founder & CEO of FRAZIYM*
