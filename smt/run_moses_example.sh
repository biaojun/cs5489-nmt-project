#!/usr/bin/env bash
set -euo pipefail

# Example SMT training pipeline for one language pair using Moses + GIZA++ + KenLM.
# Adjust MOSES_DIR, KENLM_DIR, and paths for your environment before running.

PAIR=${1:-de-en}
SRC=${PAIR%-*}
TGT=${PAIR#*-}

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/cs5489-nmt-project/processed_wmt14}
PAIR_DIR=$DATA_ROOT/$PAIR

MOSES_DIR=${MOSES_DIR:-/root/autodl-tmp/mt_tools/mosesdecoder}
KENLM_DIR=${KENLM_DIR:-/root/autodl-tmp/mt_tools/kenlm}
EXTERNAL_BIN_DIR=$MOSES_DIR/tools

SCRIPTS=$MOSES_DIR/scripts
TRAIN_SCRIPT=$SCRIPTS/training/train-model.perl
MERT_SCRIPT=$SCRIPTS/training/mert-moses.pl

WORK_DIR=$PAIR_DIR/smt
mkdir -p "$WORK_DIR"

echo "Training KenLM 5-gram LM for $PAIR..."
"$KENLM_DIR"/bin/lmplz -o 5 < "$PAIR_DIR/train.tgt" > "$WORK_DIR/lm.arpa"
"$KENLM_DIR"/bin/build_binary "$WORK_DIR/lm.arpa" "$WORK_DIR/lm.kenlm"

echo "Training Moses phrase-based SMT for $PAIR..."
"$TRAIN_SCRIPT" \
  -root-dir "$WORK_DIR" \
  -corpus "$PAIR_DIR/train" \
  -f "$SRC" -e "$TGT" \
  -alignment grow-diag-final-and \
  -reordering msd-bidirectional-fe \
  -lm 0:5:"$WORK_DIR/lm.kenlm":0 \
  -external-bin-dir "$EXTERNAL_BIN_DIR" \
  >& "$WORK_DIR/train.log"

echo "Running MERT tuning on validation set..."
"$MERT_SCRIPT" \
  "$PAIR_DIR/valid.$SRC" "$PAIR_DIR/valid.$TGT" \
  "$MOSES_DIR/bin/moses" "$WORK_DIR/model/moses.ini" \
  --mertdir "$MOSES_DIR/bin" \
  >& "$WORK_DIR/mert.log"

echo "Decoding test set..."
"$MOSES_DIR"/bin/moses -f "$WORK_DIR/mert-work/moses.ini" < "$PAIR_DIR/test.$SRC" > "$WORK_DIR/test.output"

echo "Done. SMT output at $WORK_DIR/test.output"
