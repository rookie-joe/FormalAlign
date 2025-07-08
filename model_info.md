# FormalAlign Model Information

## Model Details

**Model Name**: `formalalign_reproduce_mma_forml4_combined`  
**Base Model**: Mistral  
**Training Method**: FormalAlign  
**Model Size**: ~14GB  
**Architecture**: Transformer with CLIP verifier  

## Training Configuration

- **Generator ID**: mistral
- **Verifier ID**: mma_forml4_combined
- **Final ID**: formalalign_reproduce
- **Project Dimension**: 4096
- **Mixed Precision**: bf16
- **DeepSpeed Stage**: 1

## Model Files

```
formalalign_reproduce_mma_forml4_combined/
├── added_tokens.json
├── config.json
├── generation_config.json
├── model-00001-of-00003.safetensors (4.9GB)
├── model-00002-of-00003.safetensors (5.0GB)
├── model-00003-of-00003.safetensors (4.5GB)
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.model
├── training_args.bin
└── verifier.pth
```

## Training Data

- **MMA Dataset**: Mathematical Modeling and Analysis
- **FormL4 Dataset**: Formal Language for Mathematics
- **Alignment Strategy**: Combined misalignment detection

## Usage

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "path/to/formalalign_reproduce_mma_forml4_combined",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "path/to/formalalign_reproduce_mma_forml4_combined"
)
```

### Evaluation

```bash
# Run evaluation
bash eval_alignment.sh
```

## Performance

- **Alignment Score**: Calculated using certainty and similarity metrics
- **Evaluation Method**: Autoregressive certainty calculation (recommended)
- **Max New Tokens**: 512

## Citation

If you use this model, please cite the FormalAlign paper:

```bibtex
@article{formalalign2024,
  title={FormalAlign: Mathematical Reasoning Alignment through Formal Language},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Add your license information here] 