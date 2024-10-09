
# FormalAlign

FormalAlign is an automated framework for evaluating alignment between informal and formal mathematical statements in autoformalization tasks.

## Overview

FormalAlign addresses the challenge of ensuring semantic alignment between informal mathematical proofs and their formal counterparts. By combining **cross-entropy loss** with **contrastive learning**, our model achieves superior performance in detecting misalignments across various autoformalization benchmarks.

## Repository Structure

```
./
└── data
    ├── forml4
    │   ├── basic_misalignment.json
    │   └── random_misalignment.json
    └── minif2f
        ├── test_misalignment.json
        └── valid_misalignment.json
```

### Datasets

The repository contains misalignment data for two main datasets:

1. **FormL4**:
   - `basic_misalignment.json`: Contains basic misalignment examples from FormL4.
   - `random_misalignment.json`: Contains randomly sampled misalignment examples from FormL4.

2. **MiniF2F**:
   - `test_misalignment.json`: Contains misalignment examples from the MiniF2F test set.
   - `valid_misalignment.json`: Contains misalignment examples from the MiniF2F validation set.

## Key Features

- **Automated alignment evaluation** for autoformalization tasks.
- Combines **cross-entropy loss** and **contrastive learning** to improve model robustness.
- Outperforms GPT-4 on several autoformalization benchmarks.
- Reduces the need for **manual verification** of formal proofs.

