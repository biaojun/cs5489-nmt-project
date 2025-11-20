#!/usr/bin/env bash
set -euo pipefail

# One-time setup script to install:
# - system build tools
# - KenLM
# - Moses (with GIZA++)
# - Python libraries needed for the MT experiments
#
# Default install locations:
#   tools under /root/autodl-tmp/mt_tools
#   project under /root/autodl-tmp/cs5489-nmt-project
#
# Run as root in your container:
#   bash scripts/install_mt_env.sh

ROOT_BASE=${ROOT_BASE:-/root/autodl-tmp}
TOOLS_DIR=${TOOLS_DIR:-$ROOT_BASE/mt_tools}
PROJECT_DIR=${PROJECT_DIR:-$ROOT_BASE/cs5489-nmt-project}

echo "ROOT_BASE   = $ROOT_BASE"
echo "TOOLS_DIR   = $TOOLS_DIR"
echo "PROJECT_DIR = $PROJECT_DIR"

mkdir -p "$TOOLS_DIR"

echo "=== 1. Install system packages (requires network + apt) ==="
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake git zlib1g-dev libbz2-dev \
  libboost-all-dev automake libtool pkg-config \
  python3-pip python3-dev

echo "=== 2. Install Python packages ==="
python3 -m pip install --upgrade pip
python3 -m pip install torch datasets sentencepiece sacrebleu nltk
python3 - << 'PYEND'
import nltk
for pkg in ["wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg)
    except Exception as e:
        print("NLTK download failed for", pkg, ":", e)
PYEND

echo "=== 3. Clone and build KenLM ==="
cd "$TOOLS_DIR"
if [ ! -d kenlm ]; then
  git clone https://github.com/kpu/kenlm.git
fi
cd kenlm
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

echo "=== 4. Clone and build Moses (decoder + GIZA++) ==="
cd "$TOOLS_DIR"
if [ ! -d mosesdecoder ]; then
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
cd mosesdecoder

# Build Moses decoder (uses Boost + bjam)
if [ ! -x ./bjam ]; then
  ./bootstrap.sh
fi
./bjam -j"$(nproc)" --with-boost=/usr

# Build GIZA++ and related tools
cd tools
./build.sh

echo "=== Setup complete ==="
echo "Moses installed at: $TOOLS_DIR/mosesdecoder"
echo "KenLM installed at: $TOOLS_DIR/kenlm"
echo
echo "Next steps (from $PROJECT_DIR):"
echo "  1) Prepare data:  bash scripts/run_pipeline_de_en.sh data"
echo "  2) Train SMT:     bash smt/run_moses_example.sh de-en"
echo "  3) Train models:  python -m mt.train ..."

