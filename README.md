CS5489 Traditional Machine Translation Baselines
===============================================

This project provides a full traditional MT baseline pipeline on WMT14
for multiple language pairs (cs–en, de–en, fr–en, hi–en, ru–en), without
using large language models.

Main components:

- Arrow → plain text conversion using `datasets.load_dataset`
- Cleaning and joint SentencePiece BPE preprocessing
- SMT (Moses + GIZA++/KenLM) training and decoding
- RNN-based Seq2Seq + attention baseline (PyTorch)
- Transformer-base baseline (PyTorch)
- Evaluation with BLEU (sacrebleu) and METEOR (NLTK)
- Error analysis and report template

See `report/REPORT_TEMPLATE.md` for the structure of the final write-up,
and the `scripts/` directory for end-to-end data preparation commands.

