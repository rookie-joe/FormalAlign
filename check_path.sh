#!/bin/bash

echo "=== 模型和缓存路径检查脚本 ==="
echo ""

echo "1. Hugging Face 相关路径:"
echo "   HF_HOME: $HF_HOME"
echo "   TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "   HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo ""

echo "2. PyTorch 相关路径:"
echo "   TORCH_HOME: $TORCH_HOME"
echo ""

echo "3. 当前工作目录:"
echo "   PWD: $(pwd)"
echo ""

echo "4. 用户主目录:"
echo "   HOME: $HOME"
echo ""

echo "5. 检查 Hugging Face 缓存目录内容:"
if [ -n "$HF_HOME" ] && [ -d "$HF_HOME" ]; then
    echo "   HF_HOME 目录存在，内容如下:"
    ls -la "$HF_HOME" | head -20
    echo "   ... (显示前20行)"
else
    echo "   HF_HOME 目录不存在或未设置"
fi
echo ""

echo "6. 检查 Transformers 缓存目录:"
if [ -n "$TRANSFORMERS_CACHE" ] && [ -d "$TRANSFORMERS_CACHE" ]; then
    echo "   TRANSFORMERS_CACHE 目录存在"
    echo "   目录大小: $(du -sh "$TRANSFORMERS_CACHE" 2>/dev/null || echo '无法获取大小')"
else
    echo "   TRANSFORMERS_CACHE 目录不存在或未设置"
fi
echo ""

echo "7. 检查数据集缓存目录:"
if [ -n "$HF_DATASETS_CACHE" ] && [ -d "$HF_DATASETS_CACHE" ]; then
    echo "   HF_DATASETS_CACHE 目录存在"
    echo "   目录大小: $(du -sh "$HF_DATASETS_CACHE" 2>/dev/null || echo '无法获取大小')"
else
    echo "   HF_DATASETS_CACHE 目录不存在或未设置"
fi
echo ""


echo "=== 检查完成 ==="