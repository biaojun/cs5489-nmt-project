## 1. 数据说明

### 1.1 数据来源

- **语料库**：WMT14 多语言平行语料
- **语言对**：五个语言到英语的翻译任务
  - 捷克语→英语（cs–en）
  - 德语→英语（de–en）
  - 法语→英语（fr–en）
  - 印地语→英语（hi–en）
  - 俄语→英语（ru–en）

### 1.2 原始数据格式

- 来源：HuggingFace `wmt/wmt14` 数据集（Arrow 格式）
- 字段结构：`{"translation": {"xx": "源语言文本", "en": "英语文本"}}`
- 数据加载方式：`datasets.load_dataset("wmt/wmt14", "{src}-{tgt}")`

### 1.3 数据转换

使用 `scripts/arrow_to_txt.py` 将 Arrow 格式转换为纯文本文件：
- 输入：HuggingFace 缓存的 Arrow 文件
- 输出：每个语言对生成 `train.{src,en}`, `valid.{src,en}`, `test.{src,en}` 六个文件
- 存储路径：`processed_wmt14/{src}-{tgt}/`

------------------------------------------------------------

## 2. 数据预处理与统计

### 2.1 数据清洗

使用 `scripts/clean_corpus.py` 对原始平行语料进行清洗，规则如下：

| 清洗规则 | 说明 | 目的 |
|---------|------|------|
| 删除空行 | 移除源端或目标端为空的句对 | 避免无效训练样本 |
| 最小长度过滤 | 删除 < 3 tokens 的句子 | 过滤噪声和无意义短句 |
| 最大长度过滤 | 删除 > 256 tokens 的句子 | 控制显存占用，避免过长序列 |
| 长度比例过滤 | 删除 src/tgt 长度比 > 3 的句对 | 过滤对齐质量差的样本 |
| HTML/XML 清理 | 去除简单标签，若清理后为空则丢弃 | 移除网页爬取残留 |

### 2.2 子词分词（Subword Tokenization）

采用 **SentencePiece BPE**（Byte Pair Encoding）进行子词切分：

- **训练脚本**：`scripts/train_sentencepiece.py`
- **应用脚本**：`scripts/apply_sentencepiece.py`
- **关键参数**：
  - `vocab_size = 32000`：词表大小，平衡覆盖率和稀疏性
  - `model_type = bpe`：使用 BPE 算法
  - **共享词表**：所有语言对使用同一个 BPE 模型，便于多语言联合训练

**为什么使用 BPE？**
1. 解决 OOV（Out-of-Vocabulary）问题：将稀有词拆分为常见子词
2. 平衡词表大小和序列长度：32k 词表在翻译任务中是常用设置
3. 支持多语言：共享词表让不同语言的相似词根可以共享表示

### 2.3 多语言语料构造

使用 `scripts/build_multilingual_corpus.py` 将五个单语对合并为一个多语言数据集：

**核心设计：源端语言标签**

```
原始句子：Das ist ein Beispiel.
带标签句子：<de> Das ist ein Beispiel.
```

- 在每个源句开头添加语言标签 `<cs>`, `<de>`, `<fr>`, `<hi>`, `<ru>`
- 目标端统一为英语，不加标签
- 模型通过语言标签识别输入语言，学习"多对一"翻译

**为什么这样设计？**
- 让一个模型同时处理多种源语言
- 语言标签作为"翻译指令"，告诉模型输入是什么语言
- 共享编码器和解码器参数，实现跨语言知识迁移

### 2.4 数据统计

| 语言对 | 划分 | 句子数 | 平均源端长度 | 平均目标端长度 | 备注 |
|-------|------|--------|-------------|---------------|------|
| cs–en | train | 294,966 | 31.1 | 28.9 | 高资源 |
| de–en | train | 297,505 | 34.4 | 30.4 | 高资源 |
| fr–en | train | 297,874 | 37.6 | 30.4 | 高资源 |
| hi–en | train | 6,671 | 15.8 | 7.1 | **低资源** |
| ru–en | train | 288,795 | 37.7 | 28.4 | 高资源 |

**观察与分析**：
- **hi-en 数据量极少**（仅 6.6k 句），不到其他语言对的 2.5%，这是后续该语言对效果差的主要原因
- 法语和俄语的源端平均长度最长（~37-38 tokens），翻译难度可能更高
- 印地语的句子普遍较短，可能是数据特点或采集方式不同

------------------------------------------------------------

## 3. 模型与方法

### 3.1 RNN Seq2Seq + Attention

#### 3.1.1 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Seq2Seq 架构                           │
├─────────────────────────────────────────────────────────────┤
│  源句 → [Embedding] → [BiLSTM Encoder] → Context Vectors   │
│                              ↓                              │
│            [Luong Attention] ← Query from Decoder           │
│                              ↓                              │
│  [LSTM Decoder] → [Linear] → Softmax → 目标词概率           │
└─────────────────────────────────────────────────────────────┘
```

**组件说明**：

| 组件 | 配置 | 说明 |
|-----|------|------|
| Embedding | dim=512 | 源端和目标端共享词嵌入维度 |
| Encoder | BiLSTM, hidden=512, dropout=0.3 | 双向 LSTM，捕获上下文信息 |
| Decoder | LSTM, hidden=512 | 单向 LSTM，自回归生成 |
| Attention | Luong (dot-product) | 计算 decoder 对 encoder 各位置的关注权重 |

**Luong Attention 机制**：
$$\text{score}(h_t, \bar{h}_s) = h_t^\top \bar{h}_s$$
$$\alpha_{ts} = \frac{\exp(\text{score}(h_t, \bar{h}_s))}{\sum_{s'} \exp(\text{score}(h_t, \bar{h}_{s'}))}$$
$$c_t = \sum_s \alpha_{ts} \bar{h}_s$$

其中 $h_t$ 是 decoder 当前隐状态，$\bar{h}_s$ 是 encoder 第 $s$ 个位置的输出。

#### 3.1.2 训练策略

| 配置项 | 值 | 说明 |
|-------|-----|------|
| 优化器 | Adam | lr=1e-3 |
| Batch size | 32 | 按句子数 |
| Epochs | 20 | - |
| Teacher Forcing | 1.0 → 0.5 | 线性衰减，逐步让模型依赖自己的预测 |
| Loss | CrossEntropy | 忽略 PAD token |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸 |

**Teacher Forcing 衰减**：
- 训练初期（tf_ratio=1.0）：decoder 输入是真实的目标词，加速收敛
- 训练后期（tf_ratio=0.5）：50% 概率使用模型自己的预测，减少 exposure bias

#### 3.1.3 推理方式

- 使用 **Greedy Decoding**：每一步选择概率最大的词
- 最大生成长度：100 tokens
- 遇到 `<eos>` 停止生成

---

### 3.2 Transformer

#### 3.2.1 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer 架构                          │
├─────────────────────────────────────────────────────────────┤
│  源句 → [Embedding + Positional Encoding]                   │
│              ↓                                              │
│  [Encoder Layer × 6]                                        │
│    - Multi-Head Self-Attention                              │
│    - Feed-Forward Network                                   │
│    - Layer Norm + Residual Connection                       │
│              ↓                                              │
│  [Decoder Layer × 6]                                        │
│    - Masked Multi-Head Self-Attention                       │
│    - Cross-Attention (attend to encoder)                    │
│    - Feed-Forward Network                                   │
│              ↓                                              │
│  [Linear] → Softmax → 目标词概率                             │
└─────────────────────────────────────────────────────────────┘
```

**模型参数**：

| 参数 | 值 | 说明 |
|-----|-----|------|
| d_model | 512 | 隐藏层维度 |
| num_heads | 8 | 多头注意力的头数（每头 64 维） |
| num_encoder_layers | 6 | Encoder 层数 |
| num_decoder_layers | 6 | Decoder 层数 |
| dim_feedforward | 2048 | FFN 中间层维度 |
| dropout | 0.1 | 正则化 |

**Multi-Head Attention**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Positional Encoding**（正弦/余弦位置编码）：
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

#### 3.2.2 训练策略

| 配置项 | 值 | 说明 |
|-------|-----|------|
| 优化器 | Adam | β1=0.9, β2=0.98, ε=1e-9 |
| 学习率调度 | Warmup + Inverse Sqrt | 见下方公式 |
| Warmup steps | 4000 | 前 4000 步线性增长 |
| Batch size | 32 | 按句子数 |
| Epochs | 20 | - |
| Label Smoothing | ε=0.1 | 软化目标分布，提高泛化 |

**学习率调度（Transformer 论文原版）**：
$$lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$$

- 前 4000 步：学习率从 0 线性增长
- 之后：按 $step^{-0.5}$ 衰减

**Label Smoothing**：
- 将 one-hot 目标分布软化为 $(1-\epsilon) \cdot \text{one-hot} + \epsilon / V$
- 防止模型过度自信，提高泛化能力

#### 3.2.3 推理方式

- 使用 **Greedy Decoding**
- 起始 token：`<bos>`（ID=1）
- 终止条件：遇到 `<eos>` 或达到 max_len=100

------------------------------------------------------------

## 4. 实验配置

### 4.1 硬件与软件环境

| 项目 | 配置 |
|-----|------|
| GPU | （请填写你的 GPU 型号，如 NVIDIA RTX 3090 24GB） |
| Python | 3.12 |
| PyTorch | （请填写版本） |
| 主要依赖 | `datasets`, `sentencepiece`, `sacrebleu`, `nltk` |

### 4.2 完整训练流程

#### Step 1: 数据下载与转换
```bash
python scripts/arrow_to_txt.py \
  --cache-dir data/wmt___wmt14 \
  --save-dir processed_wmt14 \
  --mode subset
```
将 HuggingFace Arrow 格式转为纯文本。

#### Step 2: 数据清洗
```bash
python scripts/clean_corpus.py \
  --data-dir processed_wmt14 \
  --min-len 3 \
  --max-len 256 \
  --max-len-ratio 3.0
```
按照 2.1 节的规则过滤低质量句对。

#### Step 3: 训练 SentencePiece 模型
```bash
python scripts/train_sentencepiece.py \
  --data-dir processed_wmt14 \
  --vocab-size 32000 \
  --model-type bpe
```
在清洗后的数据上训练共享 BPE 词表。

#### Step 4: 应用子词分词
```bash
python scripts/apply_sentencepiece.py \
  --data-dir processed_wmt14 \
  --spm-model processed_wmt14/spm/spm.model
```
将所有文本文件转为 BPE token 序列。

#### Step 5: 构建多语言数据集
```bash
python scripts/build_multilingual_corpus.py \
  --data-dir processed_wmt14 \
  --out-name multilingual
```
合并五个语言对，添加语言标签。

#### Step 6: 训练模型
```bash
# Seq2Seq
python -m mt.train \
  --model-type seq2seq \
  --data-dir processed_wmt14 \
  --lang-pair multilingual \
  --spm-model processed_wmt14/spm/spm.model \
  --batch-size 32 \
  --epochs 20 \
  --save-dir checkpoints \
  --device cuda

# Transformer
python -m mt.train \
  --model-type transformer \
  --data-dir processed_wmt14 \
  --lang-pair multilingual \
  --spm-model processed_wmt14/spm/spm.model \
  --batch-size 32 \
  --epochs 20 \
  --save-dir checkpoints \
  --device cuda
```

#### Step 7: 评估
```bash
# 评估 Seq2Seq
python -m mt.evaluate \
  --ckpt checkpoints/multilingual.seq2seq.pt \
  --data-dir processed_wmt14 \
  --lang-pair multilingual \
  --split test \
  --device cuda \
  --out results/multilingual.seq2seq.test.json

# 评估 Transformer
python -m mt.evaluate \
  --ckpt checkpoints/multilingual.transformer.pt \
  --data-dir processed_wmt14 \
  --lang-pair multilingual \
  --split test \
  --device cuda \
  --out results/multilingual.transformer.test.json
```

------------------------------------------------------------

## 5. 评测结果

### 5.1 评测指标

| 指标 | 说明 | 计算工具 |
|-----|------|---------|
| BLEU | 基于 n-gram 精确率的翻译质量指标 | `sacrebleu` |
| METEOR | 考虑同义词和词干匹配的指标 | `nltk.translate.meteor_score` |

### 5.2 各语言对评测结果

| Model | Lang Pair | BLEU | METEOR | 分析 |
|-------|-----------|------|--------|------|
| Seq2Seq | cs-en | 4.03 | 0.147 | 中等表现 |
| Seq2Seq | de-en | 3.01 | 0.120 | 中等表现 |
| Seq2Seq | fr-en | 4.04 | 0.155 | 最佳语言对 |
| Seq2Seq | hi-en | 0.07 | 0.025 | 极低，受限于训练数据量 |
| Seq2Seq | ru-en | 2.06 | 0.107 | 较低，形态复杂 |
| Transformer | cs-en | 0.0008 | 0.029 | 严重欠拟合 |
| Transformer | de-en | 0.0009 | 0.031 | 严重欠拟合 |
| Transformer | fr-en | 0.0007 | 0.028 | 严重欠拟合 |
| Transformer | hi-en | 0.0010 | 0.030 | 严重欠拟合 |
| Transformer | ru-en | 0.0013 | 0.029 | 严重欠拟合 |

### 5.3 结果分析

**Seq2Seq 模型**：
- 在大多数语言对上取得了基本可用的翻译效果（BLEU 2-4）
- **fr-en 表现最好**：法语和英语同属印欧语系日耳曼-罗曼分支，语法结构相似
- **hi-en 表现最差**：印地语训练数据仅 6671 句，严重不足
- **ru-en 较低**：俄语形态变化丰富，且使用西里尔字母，与其他语言差异大

**Transformer 模型**：
- 所有语言对 BLEU 接近 0，出现严重的模式塌缩问题
- 从输出文件观察，模型几乎对所有输入都输出同一句话（如 "How do you think??"）
- **可能原因**：
  1. 训练不充分：batch_size=32 对 Transformer 偏小，实际有效 batch 应更大
  2. 学习率调度可能未正确生效
  3. 模型容量（6层×512维）对当前数据规模可能过大，导致过拟合到高频模式
  4. 训练可能被中断，保存的 checkpoint 是未收敛的早期版本

------------------------------------------------------------

## 6. 多语言对比分析

### 6.1 语言对难度比较

根据实验结果，语言对翻译难度排序（从易到难）：

1. **fr-en**（BLEU=4.04）：法语与英语结构相似，词序接近
2. **cs-en**（BLEU=4.03）：捷克语虽有格变化，但数据量充足
3. **de-en**（BLEU=3.01）：德语词序差异较大（动词后置）
4. **ru-en**（BLEU=2.06）：俄语形态复杂，西里尔字母系统
5. **hi-en**（BLEU=0.07）：数据严重不足，书写系统完全不同

### 6.2 数据量影响分析

| 语言对 | 训练句数 | BLEU | 每万句贡献 BLEU |
|-------|---------|------|----------------|
| fr-en | 297,874 | 4.04 | 0.136 |
| cs-en | 294,966 | 4.03 | 0.137 |
| de-en | 297,505 | 3.01 | 0.101 |
| ru-en | 288,795 | 2.06 | 0.071 |
| hi-en | 6,671 | 0.07 | 0.105 |

**观察**：hi-en 虽然总 BLEU 极低，但"单位数据贡献"并不是最低的，说明问题主要在于数据量不足，而非语言本身难度。

### 6.3 共享 BPE 词表的影响

- **优势**：允许跨语言迁移学习，相似词根可以共享表示
- **劣势**：对于书写系统差异大的语言（如印地语 Devanagari、俄语 Cyrillic），BPE 切分效率低，可能产生更长的 token 序列

------------------------------------------------------------

## 7. 错误分析

### 7.1 Seq2Seq 常见错误类型

| 错误类型 | 示例 | 分析 |
|---------|------|------|
| 重复生成 | "the the the..." | Attention 权重分布不均，decoder 陷入循环 |
| 信息丢失 | 长句中部分内容被忽略 | BiLSTM 对长距离依赖能力有限 |
| 词汇错误 | 专有名词/数字翻译错误 | BPE 将其拆分后难以正确复原 |
| 语序错误 | 主谓宾顺序与参考不符 | 源语言和目标语言语法差异 |

### 7.2 Transformer 问题诊断

当前 Transformer 输出高度重复，属于"模式塌缩"（mode collapse），需要检查：

1. **训练日志**：确认 loss 是否正常下降
2. **Checkpoint 完整性**：确认保存的是最优模型而非中途版本
3. **超参数调整**：
   - 增大 batch size（通过梯度累积）
   - 减小模型规模（如 4 层、d_model=256）
   - 检查 warmup 是否正确实现

------------------------------------------------------------

## 8. 结论与展望

### 8.1 主要结论

1. **Seq2Seq + Attention** 在多语言翻译任务上取得了基本可用的效果，尤其在高资源语言对（fr-en, cs-en）上表现较好
2. **数据量是关键因素**：hi-en 的极低性能主要归因于训练数据不足（仅 6671 句）
3. **Transformer 需要进一步调优**：当前配置下出现严重的模式塌缩，需要调整训练策略

### 8.2 改进方向

| 方向 | 具体措施 |
|-----|---------|
| 数据增强 | 对低资源语言使用回译（back-translation）增加训练数据 |
| 模型调优 | Transformer 使用更大 batch size、梯度累积、学习率调参 |
| 解码策略 | 使用 Beam Search 替代 Greedy Decoding |
| 预训练 | 利用 mBART、mT5 等预训练多语言模型进行微调 |
| 评估完善 | 增加 chrF、TER 等指标，进行人工评估 |

### 8.3 实验局限性

1. **计算资源限制**：batch size 较小，训练时间有限
2. **模型规模**：未尝试更大或更小的模型配置
3. **超参数搜索**：未进行系统的超参数调优
4. **评估方式**：仅使用自动指标，缺少人工评估

---

*报告完成日期：2025年11月30日*
