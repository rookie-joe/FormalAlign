
# FormalAlign

FormalAlign is an automated framework for evaluating alignment between informal and formal mathematical statements in autoformalization tasks.

## Overview

FormalAlign addresses the challenge of ensuring semantic alignment between informal mathematical proofs and their formal counterparts. By combining **cross-entropy loss** with **contrastive learning**, our model achieves superior performance in detecting misalignments across various autoformalization benchmarks.

## Repository Structure

```
./
└── data
    ├── forml4
    │   ├── misalignment
    │   └── alignment
    └── minif2f
        ├── misalignment
        └── alignment
```

### Datasets

The repository contains misalignment data for two main datasets:

1. **FormL4**:
   - `alignment` folder: The basic and random test sets downloaded from [FormL4](https://github.com/rookie-joe/PDA/tree/main/data/FormL4); 
      - `python format_forml4.py` was run to format the imported data files for misalignement creation.
   - `misalignment` folder
      - `formatted_basic_test.json`: Contains basic misalignment examples from FormL4.
      - `formatted_random_test.json`: Contains randomly sampled misalignment examples from FormL4.

2. **MiniF2F**:
   - `alignment` folder: the informal-formal data imported from minif2f;
   - `misalignment` folder
      - `test_misalignment.json`: Contains misalignment examples from the MiniF2F test set.
      - `valid_misalignment.json`: Contains misalignment examples from the MiniF2F validation set.

Run the following example code to replicate the creation of misalignment cases (e.g., FormL4-Basic):
```
python create_misalign.py --input_file forml4/alignment/formatted_basic_test.json --output_path forml4/misalignment --seed 42
```

## Key Features

- **Automated alignment evaluation** for autoformalization tasks.
- Combines **cross-entropy loss** and **contrastive learning** to improve model robustness.
- Outperforms GPT-4 on several autoformalization benchmarks.
- Reduces the need for **manual verification** of formal proofs.

