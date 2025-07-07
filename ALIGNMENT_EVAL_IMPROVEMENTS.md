# Alignment Evaluation Improvements

## 问题描述

原始的alignment evaluation中，certainty score的计算使用了**teacher forcing**方式，这导致了以下问题：

1. **不真实的评估**：模型在预测FL token时能看到后续的FL tokens，这不符合实际自回归生成的情况
2. **高估certainty**：Teacher forcing会导致模型confidence被高估，因为模型有"作弊"的能力
3. **Precision几乎随机**：由于评估方式不准确，导致模型区分正负样本的能力很差

## 解决方案

### 1. 自回归Certainty Score计算

**新方法**：
- 只给模型NL部分
- 让模型逐步自回归生成FL
- 计算生成的FL的sequence-level log probability

**公式**：
```
V_cer = exp(1/n * sum(log P(FL_i,j | FL_i,<j, NL_i)))
```

其中：
- `P(FL_i,j | FL_i,<j, NL_i)` 是基于NL和之前FL tokens预测当前FL token的概率
- 不能看到后续的FL tokens（真正的自回归）

### 2. 代码实现

#### 新函数：
```python
def calculate_certainty_score_autoregressive(model, tokenizer, input_ids, attention_mask, t_eoss, max_new_tokens=512):
    """
    Calculate certainty score using autoregressive generation.
    Only feeds [NL] and generates FL autoregressively.
    """
```

#### 配置选项：
- `--use_autoregressive_certainty`: 是否使用自回归方法（默认True）
- `--max_new_tokens`: 最大生成token数量（默认512）

## 使用方法

### 1. 运行评估（推荐的自回归方法）

```bash
cd alisa/FormalAlign
bash eval_alignment.sh
```

默认使用自回归方法。如果要使用旧的teacher forcing方法，修改`eval_alignment.sh`中的：
```bash
use_autoregressive_certainty=False
```

### 2. 直接Python调用

```python
# 自回归方法（推荐）
certainty_score = calculate_certainty_score_autoregressive(
    model, tokenizer, input_ids, attention_mask, t_eoss, max_new_tokens=512
)

# Teacher forcing方法（旧方法）
certainty_score = calculate_certainty_score(
    logits, input_ids, attention_mask, t_eoss
)
```

## 预期改进

1. **更真实的评估**：符合实际推理场景
2. **更准确的certainty**：不会高估模型能力
3. **更好的precision**：能更好地区分正负样本
4. **更低的certainty分数**：因为没有"作弊"能力，分数会更保守

## 性能对比

使用`test_autoregressive_certainty.py`可以比较两种方法：

```bash
python test_autoregressive_certainty.py
```

预期结果：
- 自回归方法的certainty score < Teacher forcing方法的certainty score
- 自回归方法能更好地区分正负样本

## 训练代码

训练代码保持不变，依然使用：
- Contrastive loss（对齐NL和FL embedding）
- Cross-entropy loss（语言建模）
- Teacher forcing（训练时这是标准做法）

**只有评估时改为自回归方法**。

## 论文对应

这个改进对应论文中的：
- **Section: Inference**
- **Equation (1)**: Certainty Score计算
- **要点**：评估时应该模拟真实的自回归生成过程

## 文件修改

1. `eval_alignment.py`: 新增自回归certainty score计算函数
2. `eval_alignment.sh`: 新增配置选项
3. `test_autoregressive_certainty.py`: 测试和对比脚本

## 注意事项

1. **计算开销**：自回归方法计算开销更大，但评估更准确
2. **向后兼容**：保留了旧的teacher forcing方法作为选项
3. **批处理**：自回归方法目前是逐样本计算，可能比teacher forcing慢
4. **内存使用**：自回归方法需要更多内存来存储中间状态 