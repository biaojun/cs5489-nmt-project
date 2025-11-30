CS5489 Multilingual MT Baselines
================================

This repository implements the core traditional MT baselines for the CS5489 project using **only neural methods**:

- Data preparation from local WMT14 Arrow files
- SentencePiece BPE preprocessing with shared vocabulary and language tags
- Multilingual Seq2Seq + Luong attention (PyTorch)
- Multilingual Transformer-base (PyTorch)
- Evaluation with BLEU (sacreBLEU) and METEOR (NLTK)
- Utilities for dataset statistics, result aggregation, and error analysis

Directory overview
------------------

```
analysis/                    Stats, result aggregation, loss plots, badcase sampling
   collect_results.py         Aggregate BLEU/METEOR from results/*.json
   dataset_stats.py           Compute sentence counts & avg lengths
   error_analysis.py          Sample SRC/REF/HYP triplets for manual inspection
   plot_loss_curves.py        Parse logs/*.log and draw training loss curves
data/wmt___wmt14/            Local Arrow shards (not tracked)
logs/                        Training logs (e.g., seq2seq_v2.log, transformer_v3.log)
mt/                          Training & evaluation code
   data.py                    SentencePiece-based dataset/dataloader
   train.py                   Unified trainer for Seq2Seq/Transformer
   evaluate.py                Greedy decoding + BLEU/METEOR
   models/seq2seq.py          BiLSTM encoder + Luong attention decoder
   models/transformer.py      Transformer-base implementation
processed_wmt14/             Generated plain/BPE/multilingual data
report/                      Report template and final report
scripts/                     Data preparation & automation scripts
```

Workflow summary
----------------

1. **Arrow → txt (`scripts/arrow_to_txt.py`)**  
   Reads local WMT14 Arrow files (no internet access required), writes `train/valid/test.{lang}` for all five language pairs. Supports `--mode full|subset` and `--max-train-lines`.

2. **Cleaning (`scripts/clean_corpus.py`)**  
   Removes empty lines, <3 or >256 tokens, length ratio >3×, and HTML/XML noise to produce `{split}.clean.{lang}`.

3. **SentencePiece (`scripts/train_sentencepiece.py`, `scripts/apply_sentencepiece.py`)**  
   - Train a shared 32k BPE vocabulary with language tags `<cs>`, `<de>`, `<fr>`, `<hi>`, `<ru>`.  
   - Apply the model to produce BPE tokenized `{split}.src/.tgt` for each language pair.

4. **Multilingual merge (`scripts/build_multilingual_corpus.py`)**  
   Concatenates cs/de/fr/hi/ru sources into `processed_wmt14/multilingual/{split}.src/.tgt`. Every source sentence is prefixed with its language tag so that one model can translate all five languages to English.

5. **Training**  
   - Seq2Seq: `python -m mt.train --model-type seq2seq --lang-pair multilingual ...`  
   - Transformer: `python -m mt.train --model-type transformer --lang-pair multilingual ...`

6. **Evaluation (`mt/evaluate.py`)**  
   Greedy decoding on the multilingual test set with sacreBLEU + METEOR outputs. A `.txt` file containing `SRC/REF/HYP` triples is stored for error analysis.

Multilingual models
-------------------

### Seq2Seq + Attention

- BiLSTM encoder (hidden=512, dropout=0.3)  
- Luong attention with explicit handling of the 2× hidden encoder dimension  
- Single-layer LSTM decoder with teacher forcing ratio linearly decaying 1.0 → 0.5  
- Optimizer: Adam(lr=1e-3), batch_size=64 sentences, epochs=20 (tune as needed)  
- Checkpoint: `checkpoints/multilingual.seq2seq.pt`

### Transformer-base

- 6 encoder layers / 6 decoder layers, d_model=512, nhead=8, FFN=2048, dropout=0.1  
- Optimizer: Adam(β1=0.9, β2=0.98, eps=1e-9) with Noam LR schedule (warmup=4000)  
- Label smoothing 0.1, batch size ≈4096 tokens (adjust for GPU memory)  
- Checkpoint: `checkpoints/multilingual.transformer.pt`

One-click scripts
-----------------

- **Multilingual end-to-end pipeline (cs/de/fr/hi/ru → en)**  
  ```bash
  bash scripts/run_multilingual_pipeline.sh subset   # or 'full'
  ```
  Steps: Arrow→txt, cleaning, SentencePiece training/apply, multilingual merge, Seq2Seq training/eval, Transformer training/eval.

- **Single-language pipeline (optional ablation, e.g., de→en)**  
  ```bash
  bash scripts/run_pipeline_de_en.sh subset
  ```

Evaluation & analysis
---------------------

- `mt/evaluate.py` (with `--lang-pair multilingual`)  
  ```bash
  python -m mt.evaluate \
    --ckpt checkpoints/multilingual.seq2seq.pt \
    --lang-pair multilingual \
    --data-dir /root/autodl-tmp/processed_wmt14 \
    --split test \
    --device cuda \
    --out results/multilingual.seq2seq.test.json
  ```
- `analysis/collect_results.py` aggregates metrics from `results/*.json` into the required BLEU table.  
- `analysis/error_analysis.py` samples badcases by combining reference/test outputs from Seq2Seq and Transformer models.
- `analysis/plot_loss_curves.py` parses `logs/seq2seq_v2.log` and `logs/transformer_v3.log` to produce separate training loss plots `analysis/seq2seq_v2_loss.png` and `analysis/transformer_v3_loss.png`.

Tips & notes
------------

- Ensure SentencePiece training uses the same cleaned corpora that you plan to tokenize; the provided script automatically injects language tags into the vocabulary.
- The multilingual dataset lives in `processed_wmt14/multilingual`. When passing `--lang-pair multilingual`, both training and evaluation scripts read from this folder.
- Reduce training time by setting `--mode subset --max-train-lines 50000` during Arrow→txt conversion. Increase the value for higher final BLEU.
- Before reporting results, always run `mt/evaluate.py` on the multilingual test set to obtain corpus BLEU/METEOR along with qualitative outputs.
- Refer to `report/REPORT_TEMPLATE.md` for the expected structure of the final write-up (data description, preprocessing, model details, BLEU/METEOR tables, multilingual analysis, error cases, conclusions).

With these scripts you can build, train, and evaluate multilingual Seq2Seq and Transformer baselines that translate Czech, German, French, Hindi, and Russian into English without relying on Moses/SMT components.
