# XorZen Advanced Techniques Documentation

The `ult` (Ultimate) sub-library implements the cutting-edge training strategies that allow XorZen models to achieve high performance with a fraction of the parameter count of traditional models.

---

## 1. Multi-Teacher Distillation

### `xorzen_distill.py`
The `DistillationMaster` class is responsible for transferring knowledge from large, pre-trained teacher models (e.g., GPT-2 XL, GPT-J) into XorZen's lightweight student models.
- **Soft Target Distillation**: Uses a temperature-scaled KL divergence loss to capture the full probability distribution of the teacher, which contains more information than simple hard labels.
- **Vocabulary Alignment**: Automatically handles differences in vocabulary size between the teacher and the student by aligning the overlapping token IDs.
- **Alpha Balancing**: Allows for a weighted blend between the distillation loss and the standard cross-entropy (hard) loss.

### `DataAugmenter`
Works alongside distillation to increase dataset diversity through on-the-fly text perturbations, ensuring the student model generalizes better.

---

## 2. Ultra-Fast Transfer Learning

### `xorzen_ultimate.py`
This module implements a "Transfer Learning on Steroids" approach, designed to jumpstart a 1M parameter model to beat models many times its size.

#### `SuperFastTransfer`
Instead of training from scratch, it extracts pre-trained knowledge directly from a teacher:
1. **Embedding Transfer**: Instantly populates the student's vocabulary space with teacher embeddings.
2. **Expert Seeding**: Seeds the MoE (Mixture of Experts) fabric by adapting MLP weights from the teacher's layers. Each teacher layer can spawn multiple student experts with slight noise to encourage specialization.
3. **Smart Adaptation**: Uses orthogonal projections and noise-injection to adapt teacher weights to the student's specific architecture (e.g., hidden dimension mismatches).

#### `SmartTrainer`
A high-level trainer specifically designed for the Ultimate pipeline. It integrates the transfer learning phase with a subsequent refinement phase to solidify the inherited knowledge.

---

## 3. High-Level API
The `__init__.py` file exposes these classes as the flagship training methods for the XorZen ecosystem, providing a one-command path to "superintelligence" training.

---
*Back to [README.md](../README.md)*
