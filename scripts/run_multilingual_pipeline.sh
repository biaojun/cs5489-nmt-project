#!/usr/bin/env bash
set -euo pipefail

# End-to-end multilingual pipeline (cs/de/fr/hi/ru -> en):
#   1. Arrow -> txt (full or subset)
#   2. Cleaning
#   3. Train & apply SentencePiece
#   4. Merge into multilingual dataset with language tags
#   5. Train Seq2Seq and Transformer on merged data
#   6. Evaluate on multilingual test set
#
# Usage:
#   bash scripts/run_multilingual_pipeline.sh subset   # or 'full'

MODE=${1:-subset}
MAX_TRAIN_LINES=${MAX_TRAIN_LINES:-50000}

PROJECT_ROOT=/root/autodl-tmp/cs5489-nmt-project
CACHE_DIR=$PROJECT_ROOT/data/wmt___wmt14
PROC_DIR=$PROJECT_ROOT/processed_wmt14
SPM_MODEL=$PROC_DIR/spm/spm.model
MULTI_DIR=$PROC_DIR/multilingual

cd "$PROJECT_ROOT"

echo "=== 1. Arrow -> txt (${MODE}) ==="
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

echo "=== 5. Build multilingual dataset ==="
python scripts/build_multilingual_corpus.py \
  --data-dir "$PROC_DIR" \
  --out-name multilingual

echo "=== 6. Train multilingual Seq2Seq ==="
python -m mt.train \
  --model-type seq2seq \
  --data-dir  "$PROC_DIR" \
  --lang-pair multilingual \
  --spm-model "$SPM_MODEL" \
  --batch-size 64 \
  --epochs 20 \
  --save-dir checkpoints \
  --device cuda

echo "=== 7. Train multilingual Transformer ==="
python -m mt.train \
  --model-type transformer \
  --data-dir  "$PROC_DIR" \
  --lang-pair multilingual \
  --spm-model "$SPM_MODEL" \
  --batch-size 4096 \
  --epochs 20 \
  --save-dir checkpoints \
  --device cuda

mkdir -p results

echo "=== 8. Evaluate Seq2Seq on multilingual test ==="
python -m mt.evaluate \
  --ckpt checkpoints/multilingual.seq2seq.pt \
  --data-dir "$PROC_DIR" \
  --lang-pair multilingual \
  --split test \
  --device cuda \
  --out results/multilingual.seq2seq.test.json

echo "=== 9. Evaluate Transformer on multilingual test ==="
python -m mt.evaluate \
  --ckpt checkpoints/multilingual.transformer.pt \
  --data-dir "$PROC_DIR" \
  --lang-pair multilingual \
  --split test \
  --device cuda \
  --out results/multilingual.transformer.test.json

echo "=== Multilingual pipeline completed ==="
