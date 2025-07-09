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
    - `train_clip`: 原版（with minimal change）
    - `eval_clip+cert`: alignnment score = embedding simlarity + certainty
        - build model tokenizer: padding = right, load model tokenizer: padding = left -> right (`utils/models.py/load_model`)。评估与训练保持一致。

```
DataArguments(data_dir='../data/clean_set/minimal.jsonl', target_set='test', generator_id='mistral', data_id='clean_set', verifier_id='mma_forml4_combined_v2', verifier_output_dir='/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/eval_results/mistral_mma_forml4_combined_v2/clean_set_minimal', generator_metric_dir='eval_results/gsm8k/generator_with_verifier', easy=True)
```

```
        # check why : 
File "/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/utils/metrics.py", line 324, in get_metric
                    corrs = np.where(gts, preds, ~preds)
                ValueError: operands could not be broadcast together with shapes (2160,) (10,) (10,)
```