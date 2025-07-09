# Reproduction log:

- original script with minimal changes:
    - `train_clip`：原版（with minimal change）
    - `eval_clip`: 原版，只用了cosine simlarity来作为alignment score？i.e., only clip

- changed v1: 用_alignment结尾的
    - `train_alignment`:严格对照论文重写了很多custom code （比如提取embedding simlarity，计算log prob和loss），而不是用已有的package
    - `eval_alignment`： 同上
    - `utils_alignment`：配合上面做了些小修改
    - `/checkpoints/mistral/formalalign_reproduce_mma_forml4_combined_v1`: 用`train_alignment`训练的模型，过拟合。失败。

- reproduction v2:
    - 
