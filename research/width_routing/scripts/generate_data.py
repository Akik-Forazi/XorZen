#!/usr/bin/env python3
"""
Generate copy-and-transform dataset with two difficulty levels.

Task: Given a 16-token sequence, predict the next 3 tokens.
- Easy: next 3 tokens = copy of tokens at positions 2,3,4 (identity)
- Hard: next 3 tokens = reverse of tokens at positions 2,3,4

The model sees the full 16-token input and must predict tokens at positions 5,6,7.
The "answer" is encoded in positions 2,3,4 — the model must learn the transform.

Vocabulary: 20 tokens (IDs 1-20, 0 reserved for padding)
Sequence length: 16 (positions 0-15)
Predict positions: 5, 6, 7
Source positions: 2, 3, 4

Difficulty is NOT identifiable from:
- A single token (both easy/hard use same token distribution)
- Sequence length (all sequences are length 16)
- Label frequency (both produce 3 tokens from same source)
- Position (source is always positions 2,3,4)

Difficulty IS identifiable from the relationship between source and target:
- Easy: target[5]=source[2], target[6]=source[3], target[7]=source[4]
- Hard: target[5]=source[4], target[6]=source[3], target[7]=source[2]

The model must learn which transform to apply. This requires understanding
the spatial relationship, not just memorizing token frequencies.
"""
import torch
import numpy as np
import json
from pathlib import Path

VOCAB_SIZE = 20  # tokens 1-20 (0 = pad)
SEQ_LEN = 16
SOURCE_POS = [2, 3, 4]
TARGET_POS = [5, 6, 7]


def generate_example(difficulty: str, rng: np.random.Generator) -> dict:
    """Generate one example.

    Args:
        difficulty: 'easy' or 'hard'
        rng: numpy random generator

    Returns:
        dict with 'input_ids', 'labels', 'difficulty'
    """
    # Random tokens for the full sequence (positions 0-15)
    tokens = rng.integers(1, VOCAB_SIZE + 1, size=SEQ_LEN).tolist()

    # Source tokens at positions 2, 3, 4
    source = [tokens[SOURCE_POS[0]], tokens[SOURCE_POS[1]], tokens[SOURCE_POS[2]]]

    if difficulty == 'easy':
        # Easy: copy source to target positions (identity)
        tokens[TARGET_POS[0]] = source[0]
        tokens[TARGET_POS[1]] = source[1]
        tokens[TARGET_POS[2]] = source[2]
    elif difficulty == 'hard':
        # Hard: reverse source at target positions
        tokens[TARGET_POS[0]] = source[2]
        tokens[TARGET_POS[1]] = source[1]
        tokens[TARGET_POS[2]] = source[0]
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    # Labels: the model should predict positions 5,6,7 from positions 0-4
    # We use standard LM labels (same as input_ids, shifted in loss)
    return {
        'input_ids': tokens,
        'labels': tokens,  # will be shifted by model
        'difficulty': difficulty,
    }


def generate_dataset(n_easy: int, n_hard: int, seed: int) -> list:
    """Generate a mixed dataset."""
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n_easy):
        examples.append(generate_example('easy', rng))
    for _ in range(n_hard):
        examples.append(generate_example('hard', rng))
    # Shuffle
    rng.shuffle(examples)
    return examples


def to_tensors(examples: list):
    """Convert to batched tensors."""
    input_ids = torch.tensor([e['input_ids'] for e in examples], dtype=torch.long)
    labels = torch.tensor([e['labels'] for e in examples], dtype=torch.long)
    difficulties = [e['difficulty'] for e in examples]
    return input_ids, labels, difficulties


def main():
    output_dir = Path(__file__).resolve().parents[0] / 'data'
    output_dir.mkdir(exist_ok=True)

    # Generate train/val/test
    train = generate_dataset(1000, 1000, seed=42)
    val = generate_dataset(200, 200, seed=123)
    test = generate_dataset(200, 200, seed=456)

    # Save as JSON
    for name, data in [('train', train), ('val', val), ('test', test)]:
        path = output_dir / f'{name}.json'
        with open(path, 'w') as f:
            json.dump(data, f)
        n_easy = sum(1 for e in data if e['difficulty'] == 'easy')
        n_hard = sum(1 for e in data if e['difficulty'] == 'hard')
        print(f"  {name}: {len(data)} examples ({n_easy} easy, {n_hard} hard) → {path}")

    # Print sample examples
    print("\n  Sample easy example:")
    e = train[0]
    print(f"    input_ids: {e['input_ids']}")
    print(f"    source (pos 2,3,4): {e['input_ids'][2]}, {e['input_ids'][3]}, {e['input_ids'][4]}")
    print(f"    target (pos 5,6,7): {e['input_ids'][5]}, {e['input_ids'][6]}, {e['input_ids'][7]}")
    print(f"    difficulty: {e['difficulty']}")

    # Find a hard example
    for e in train:
        if e['difficulty'] == 'hard':
            print(f"\n  Sample hard example:")
            print(f"    input_ids: {e['input_ids']}")
            print(f"    source (pos 2,3,4): {e['input_ids'][2]}, {e['input_ids'][3]}, {e['input_ids'][4]}")
            print(f"    target (pos 5,6,7): {e['input_ids'][5]}, {e['input_ids'][6]}, {e['input_ids'][7]}")
            print(f"    difficulty: {e['difficulty']}")
            break

    # Verify no shortcut: token frequency should be identical easy vs hard
    print("\n  Token frequency check (no shortcut):")
    train_easy = [e for e in train if e['difficulty'] == 'easy']
    train_hard = [e for e in train if e['difficulty'] == 'hard']
    easy_tokens = [t for e in train_easy for t in e['input_ids']]
    hard_tokens = [t for e in train_hard for t in e['input_ids']]
    easy_freq = np.bincount(easy_tokens, minlength=VOCAB_SIZE + 1)
    hard_freq = np.bincount(hard_tokens, minlength=VOCAB_SIZE + 1)
    easy_freq = easy_freq / easy_freq.sum()
    hard_freq = hard_freq / hard_freq.sum()
    max_freq_diff = np.abs(easy_freq - hard_freq).max()
    print(f"    max token frequency diff (easy vs hard): {max_freq_diff:.6f}")
    print(f"    (should be small — no token-ID shortcut)")


if __name__ == '__main__':
    print("Generating copy-and-transform dataset...")
    main()
