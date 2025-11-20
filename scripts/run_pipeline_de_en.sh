#!/usr/bin/env bash
set -euo pipefail

# End-to-end example pipeline for de-en:
#  - Arrow -> txt
#  - cleaning
#  - SentencePiece BPE train + apply
#  - Seq2Seq training
#  - Transformer training
#  - Evaluation (BLEU + METEOR) on test set
#
# Usage (from project root):
#   bash scripts/run_pipeline_de_en.sh full      # use all train data
#   bash scripts/run_pipeline_de_en.sh subset    # use subset (default 50k)

MODE=${1:-subset}        # full or subset
MAX_TRAIN_LINES=${MAX_TRAIN_LINES:-50000}

PROJECT_ROOT=/root/autodl-tmp/cs5489-nmt-project
CACHE_DIR=$PROJECT_ROOT/data/wmt___wmt14
PROC_DIR=$PROJECT_ROOT/processed_wmt14
SPM_MODEL=$PROC_DIR/spm/spm.model

cd "$PROJECT_ROOT"

echo "=== 1. Arrow -> plain text (${MODE}) ==="
python scripts/arrow_to_txt.py \
  --cache-dir "$CACHE_DIR" \
  --save-dir  "$PROC_DIR" \
  --mode "$MODE" \
  --max-train-lines "$MAX_TRAIN_LINES"

echo "=== 2. Clean corpora ==="
python scripts/clean_corpus.py \
  --data-dir "$PROC_DIR" \
  --min-len 3 --max-len 256 --max-len-ratio 3.0

echo "=== 3. Train SentencePiece (joint BPE) ==="
python scripts/train_sentencepiece.py \
  --data-dir "$PROC_DIR" \
  --vocab-size 32000 \
  --model-type bpe

echo "=== 4. Apply SentencePiece ==="
python scripts/apply_sentencepiece.py \
  --data-dir "$PROC_DIR" \
  --spm-model "$SPM_MODEL"

echo "=== 5. Train Seq2Seq (de-en) ==="
python -m mt.train \
  --model-type seq2seq \
  --data-dir  "$PROC_DIR" \
  --lang-pair de-en \
  --spm-model "$SPM_MODEL" \
  --batch-size 64 \
  --epochs 20 \
  --save-dir checkpoints \
  --device cuda

echo "=== 6. Train Transformer-base (de-en) ==="
python -m mt.train \
  --model-type transformer \
  --data-dir  "$PROC_DIR" \
  --lang-pair de-en \
  --spm-model "$SPM_MODEL" \
  --batch-size 4096 \
  --epochs 20 \
  --save-dir checkpoints \
  --device cuda

mkdir -p results

echo "=== 7. Evaluate Seq2Seq (BLEU & METEOR) on test ==="
python -m mt.evaluate \
  --ckpt checkpoints/de-en.seq2seq.pt \
  --data-dir "$PROC_DIR" \
  --lang-pair de-en \
  --split test \
  --device cuda \
  --out results/de-en.seq2seq.test.json

echo "=== 8. Evaluate Transformer (BLEU & METEOR) on test ==="
python -m mt.evaluate \
  --ckpt checkpoints/de-en.transformer.pt \
  --data-dir "$PROC_DIR" \
  --lang-pair de-en \
  --split test \
  --device cuda \
  --out results/de-en.transformer.test.json

echo "=== Pipeline for de-en completed ==="
echo "Checkpoints: $PROJECT_ROOT/checkpoints"
echo "Results:     $PROJECT_ROOT/results"

