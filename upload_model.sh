#!/bin/bash

# 模型上传脚本
# 上传FormalAlign模型到Hugging Face Hub

# 设置模型路径
MODEL_PATH="/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/checkpoints/mistral/formalalign_reproduce_mma_forml4_combined"

# 设置Hugging Face仓库名称（需要修改为你的用户名）
HF_USERNAME="alisa-yingjia-wan"  # 请替换为你的Hugging Face用户名
REPO_NAME="formalalign_v1-mistral-mma-forml4"

# 完整的仓库路径
FULL_REPO_NAME="${HF_USERNAME}/${REPO_NAME}"

echo "准备上传模型到: ${FULL_REPO_NAME}"
echo "模型路径: ${MODEL_PATH}"
echo "模型大小: $(du -sh ${MODEL_PATH})"

# 检查是否已登录Hugging Face
if ! huggingface-cli whoami &> /dev/null; then
    echo "请先登录Hugging Face:"
    echo "huggingface-cli login"
    exit 1
fi

# 创建README文件
cat > README.md << 'EOF'
# FormalAlign: Mistral-based Mathematical Alignment Model

This model is trained using the FormalAlign approach for mathematical reasoning alignment.

## Model Details
- **Base Model**: Mistral
- **Training Method**: FormalAlign with MMA and FormL4 datasets
- **Model Size**: ~14GB
- **Architecture**: Transformer with CLIP verifier

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your_username/formalalign-mistral-mma-forml4")
tokenizer = AutoTokenizer.from_pretrained("your_username/formalalign-mistral-mma-forml4")
```

## Training Data
- MMA (Mathematical Modeling and Analysis)
- FormL4 (Formal Language for Mathematics)

## Citation
If you use this model, please cite the FormalAlign paper.
EOF

# 上传模型
echo "开始上传模型..."
huggingface-cli upload ${FULL_REPO_NAME} ${MODEL_PATH} --repo-type model

# 上传README
echo "上传README文件..."
huggingface-cli upload ${FULL_REPO_NAME} README.md --repo-type model

echo "上传完成！"
echo "模型地址: https://huggingface.co/${FULL_REPO_NAME}" 