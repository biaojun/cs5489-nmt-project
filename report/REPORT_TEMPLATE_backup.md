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
  - 输出到 `/home/ljb/MT/cs5489-nmt-project/processed_wmt14/{src}-{tgt}/train.{src,tgt}, valid.{src,tgt}, test.{src,tgt}`

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
    - `/home/ljb/MT/cs5489-nmt-project/processed_wmt14/{src}-{tgt}/{split}.src`（源语言 BPE）
    - `/home/ljb/MT/cs5489-nmt-project/processed_wmt14/{src}-{tgt}/{split}.tgt`（英文 BPE）

- 多语言语料构造（参考 `scripts/build_multilingual_corpus.py`）：
  - 将 {cs,de,fr,hi,ru}→en 的单语对拼接成一个多语数据集 `multilingual`
  - 在源句开头添加语言标签，例如 `<cs> ...`, `<de> ...`，用于标记源语言
  - 多语数据存放在：`/home/ljb/MT/cs5489-nmt-project/processed_wmt14/multilingual/{train,valid,test}.{src,tgt}`

- 数据统计（训练集）：
| Lang Pair | Split | #Sentences | Avg Src Tokens | Avg Tgt Tokens |
| --------- | ----- | ---------- | -------------- | -------------- |
| cs–en     | train | 294966     | 31.1           | 28.9           |
| de–en     | train | 297505     | 34.4           | 30.4           |
| fr–en     | train | 297874     | 37.6           | 30.4           |
| hi–en     | train | 6671       | 15.8           | 7.1            |
| ru–en     | train | 288795     | 37.7           | 28.4           |


**3. 模型与方法**

3.1 RNN Seq2Seq + Attention

- 实现文件：`mt/models/seq2seq.py`, `mt/train.py`
- 模型结构：
  - Encoder：BiLSTM，hidden=512，dropout=0.3
  - Decoder：单层 LSTM，hidden=512
  - Attention：Luong 注意力
  - Embedding 维度：512
- 训练配置：
  - 优化器：Adam(lr=1e-3)
  - batch_size：32（按句子数，本项目实际使用值）
  - epochs：20
  - teacher forcing 比例：从 1.0 线性下降到 0.5
  - Loss：交叉熵（忽略 PAD）
  - 多语言设置：源端带语言标签 `<cs>/<de>/<fr>/<hi>/<ru>`，目标端统一为英语

3.2 Transformer

- 实现文件：`mt/models/transformer.py`, `mt/train.py`
- 模型结构：
  - encoder layers = 6
  - decoder layers = 6
  - d_model = 512
  - num_heads = 8
  - feedforward dim = 2048
  - dropout = 0.1
- 训练配置（按本项目实际实现）：
  - batch_size：32（按句子数）
  - epochs：20
  - 优化器：Adam(β1=0.9, β2=0.98, eps=1e-9)
  - 初始学习率：1.0，与 `d_model` 和 `warmup` 共同决定有效学习率
  - 学习率调度：Transformers 论文中的 schedule
    $$lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$$
    其中 warmup steps = 4000
  - Loss：带 label smoothing 的交叉熵，平滑系数 $\epsilon = 0.1$

**4. 实验配置**

- 硬件环境（GPU 型号、显存等）
- 软件环境：Python / PyTorch 版本、`datasets`, `sentencepiece`, `sacrebleu`, `nltk` 等版本
- 训练命令示例（每个语言对、每个模型各给出一条）：
  - 数据准备（在项目根目录 `/home/ljb/MT/cs5489-nmt-project` 下）：
    - `python scripts/arrow_to_txt.py --cache-dir data/wmt___wmt14 --save-dir processed_wmt14 --mode subset`
    - `python scripts/clean_corpus.py --data-dir processed_wmt14 --min-len 3 --max-len 256 --max-len-ratio 3.0`
    - `python scripts/train_sentencepiece.py --data-dir processed_wmt14 --vocab-size 32000 --model-type bpe`
    - `python scripts/apply_sentencepiece.py --data-dir processed_wmt14 --spm-model processed_wmt14/spm/spm.model`
    - `python scripts/build_multilingual_corpus.py --data-dir processed_wmt14 --out-name multilingual`
  - 多语言 Seq2Seq：
    - `python -m mt.train --model-type seq2seq --data-dir processed_wmt14 --lang-pair multilingual --spm-model processed_wmt14/spm/spm.model --batch-size 32 --epochs 20 --save-dir checkpoints --device cuda --no-multi-gpu`
  - 多语言 Transformer：
    - `python -m mt.train --model-type transformer --data-dir processed_wmt14 --lang-pair multilingual --spm-model processed_wmt14/spm/spm.model --batch-size 32 --epochs 20 --save-dir checkpoints --device cuda --no-multi-gpu`
  - 多语言评估：
    - `python -m mt.evaluate --ckpt checkpoints/multilingual.seq2seq.pt --data-dir processed_wmt14 --lang-pair multilingual --split test --device cuda --out results/multilingual.seq2seq.test.json`
    - `python -m mt.evaluate --ckpt checkpoints/multilingual.transformer.pt --data-dir processed_wmt14 --lang-pair multilingual --split test --device cuda --out results/multilingual.transformer.test.json`

**5. 评测结果（BLEU & METEOR）**

使用 `mt/evaluate.py` 在 test 集上计算 BLEU（sacrebleu）与 METEOR（NLTK）。本项目当前主要完成了多语言（cs/de/fr/hi/ru→en）联合模型的实验，结果如下：

| Model       | Lang Pair | BLEU   | METEOR |
|-------------|-----------|--------|--------|
| seq2seq     | cs-en     | 4.0291 | 0.1474 |
| seq2seq     | de-en     | 3.0079 | 0.1203 |
| seq2seq     | fr-en     | 4.0391 | 0.1545 |
| seq2seq     | hi-en     | 0.0722 | 0.0253 |
| seq2seq     | ru-en     | 2.0629 | 0.1072 |
| transformer | cs-en     | 0.0008 | 0.0293 |
| transformer | de-en     | 0.0009 | 0.0305 |
| transformer | fr-en     | 0.0007 | 0.0280 |
| transformer | hi-en     | 0.0010 | 0.0300 |
| transformer | ru-en     | 0.0013 | 0.0294 |


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

