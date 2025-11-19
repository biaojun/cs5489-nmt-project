传统机器翻译基线实验报告模板
==============================

本报告模板对应 CS5489 项目要求，覆盖数据说明、模型设计、训练配置、结果与分析等内容。你可以在完成实验后，将具体数值和案例填入本模板。

------------------------------------------------------------

**1. 数据说明**

- 数据来源：WMT14 多语言平行语料（cs–en, de–en, fr–en, hi–en, ru–en）
- 原始格式：HuggingFace `wmt/wmt14` Arrow 文件，字段：
  - `{"translation": {"xx": "...", "en": "..."}}`
- Arrow → txt 转换方式：
  - 使用 `datasets.load_dataset("wmt/wmt14", "{src}-{tgt}")`
  - 参考脚本：`scripts/arrow_to_txt.py`
  - 输出到 `/root/autodl-tmp/processed_wmt14/{src}-{tgt}/train.{src,tgt}, valid.{src,tgt}, test.{src,tgt}`

**2. 数据预处理与统计**

- 清洗规则（参考 `scripts/clean_corpus.py`）：
  - 删除空行
  - 删除过短句子（<3 tokens）
  - 删除过长句子（>256 tokens）
  - 删除 source / target 长度差距 >3 倍的样本
  - 去除简单 HTML/XML 标签，对应样本若为空则丢弃
- Tokenization：
  - 使用 SentencePiece BPE（共享词表）
  - `vocab_size = 32000, model_type = bpe`
  - 参考脚本：`scripts/train_sentencepiece.py`, `scripts/apply_sentencepiece.py`
  - 最终输出格式：
    - `/root/autodl-tmp/processed_wmt14/{src}-{tgt}/{split}.src`（源语言 BPE）
    - `/root/autodl-tmp/processed_wmt14/{src}-{tgt}/{split}.tgt`（英文 BPE）

在此处给出数据统计表（可用 `analysis/dataset_stats.py` 生成）：

| Lang Pair | Split | #Sentences | Avg Src Tokens | Avg Tgt Tokens |
| --------- | ----- | ---------- | -------------- | -------------- |
| cs–en     | train |            |                |                |
| de–en     | train |            |                |                |
| fr–en     | train |            |                |                |
| hi–en     | train |            |                |                |
| ru–en     | train |            |                |                |

（可按需要补充 valid/test 统计）

**3. 模型与方法**

3.1 SMT（Moses）

- 对齐：
  - 使用 Moses 自带 GIZA++ 进行双向词对齐 + grow-diag-final-and
- 语言模型：
  - 使用 KenLM 训练 5-gram LM（目标语言为 English）
  - 参考命令：`lmplz -o 5` + `build_binary`
- 短语表与重排序：
  - Moses `train-model.perl` 自动生成 phrase table 和 reordering table（`msd-bidirectional-fe`）
- 调优：
  - 使用 MERT 在 validation 集上调优 log-linear 模型权重
- 解码：
  - 使用 Moses decoder 翻译 test 集
- 参考脚本：`smt/run_moses_example.sh`

3.2 RNN Seq2Seq + Attention

- 实现文件：`mt/models/seq2seq.py`, `mt/train.py`
- 模型结构：
  - Encoder：BiLSTM，hidden=512，dropout=0.3
  - Decoder：单层 LSTM，hidden=512
  - Attention：Luong 注意力
  - Embedding 维度：512
- 训练配置：
  - 优化器：Adam(lr=1e-3)
  - batch_size：64（按句子数）
  - epochs：20
  - teacher forcing 比例：从 1.0 线性下降到 0.5
  - Loss：交叉熵（忽略 PAD）

3.3 Transformer-base

- 实现文件：`mt/models/transformer.py`, `mt/train.py`
- 结构设置（Vaswani Transformer-base）：
  - encoder layers = 6
  - decoder layers = 6
  - d_model = 512
  - num_heads = 8
  - ffn_dim = 2048
  - dropout = 0.1
  - label_smoothing = 0.1
- 训练配置：
  - batch_size ≈ 4096 tokens（可通过调整 DataLoader 或梯度累积实现）
  - optimizer：Adam(β1=0.9, β2=0.98, eps=1e-9)
  - learning rate：`d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})`
  - warmup steps = 4000

**4. 实验配置**

- 硬件环境（GPU 型号、显存等）
- 软件环境：Python / PyTorch 版本、`datasets`, `sentencepiece`, `sacrebleu`, `nltk` 等版本
- 训练命令示例（每个语言对、每个模型各给出一条）：
  - 数据准备：
    - `python scripts/arrow_to_txt.py`
    - `python scripts/clean_corpus.py`
    - `python scripts/train_sentencepiece.py`
    - `python scripts/apply_sentencepiece.py`
  - SMT：
    - `bash smt/run_moses_example.sh de-en`
  - Seq2Seq：
    - `python -m mt.train --model-type seq2seq --lang-pair de-en --batch-size 64 --epochs 20`
  - Transformer：
    - `python -m mt.train --model-type transformer --lang-pair de-en --batch-size 4096`

**5. 评测结果（BLEU & METEOR）**

使用 `mt/evaluate.py` 在 test 集上计算 BLEU（sacrebleu）与 METEOR（NLTK），并将结果汇总成表：

| Model            | cs→en | de→en | fr→en | hi→en | ru→en | Avg |
| ---------------- | ----- | ----- | ----- | ----- | ----- | --- |
| SMT (BLEU)       |       |       |       |       |       |     |
| Seq2Seq (BLEU)   |       |       |       |       |       |     |
| Transformer (BLEU) |     |       |       |       |       |     |

可在附录中给出对应的 METEOR 表格，或者将 BLEU / METEOR 同时汇总。

**6. 多语言对比分析**

- 比较不同语言对的难度（如 hi–en / ru–en 通常更难）
- 分析 SMT vs Seq2Seq vs Transformer 的整体趋势：
  - SMT 在高资源语言对（de–en, fr–en）表现如何？
  - Seq2Seq 是否在长句、形态丰富语言上有优势/劣势？
  - Transformer 是否在所有语言对上都明显优于 RNN？
- 讨论 BPE 共享词表对多语言的影响（尤其是 hi/ru 等与西欧语言差异较大的情况）

**7. 错误分析（Badcase）**

- 使用 `analysis/error_analysis.py` 从 test 集中抽取每个语言对 10 个样例：
  - source
  - reference
  - SMT output
  - Seq2Seq output
  - Transformer output
  - 错误类型标签（可手动标注：缺词 / 错词 / 乱序 / 语义错误 / 其他）
- 针对典型错误进行分类讨论：
  - SMT：常见的短语边界错误、词形变化错误、稀有词翻译失败
  - Seq2Seq：长句信息丢失、重复生成、对齐不稳定
  - Transformer：对长距离依赖的处理、专有名词/数字翻译、语序自然度

**8. 结论**

- 总结不同方法在多语言 WMT14 上的整体表现：
  - 哪种方法在平均 BLEU/METEOR 上表现最好？
  - 是否存在某些语言对上传统 SMT 仍具有竞争力？
- 讨论可能的改进方向：
  - 更强的语言模型或 n-gram/backoff 设计
  - 更深/正则化更好的 Seq2Seq/Transformer
  - 多语种联合训练 / 共享编码器等
- 指出本实验的局限性（训练时间、模型容量、没有使用更大规模数据等），并说明如果进一步扩展可以做哪些工作。

