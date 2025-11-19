"""
Aggregate BLEU / METEOR scores from JSON files into a markdown table.

Expected input JSON structure:
{
  "model": "seq2seq" | "transformer" | "smt",
  "lang_pair": "de-en",
  "split": "test",
  "bleu": 27.3,
  "meteor": 0.32
}
"""

import argparse
import glob
import json
from collections import defaultdict


LANG_PAIRS = ["cs-en", "de-en", "fr-en", "hi-en", "ru-en"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True, help="glob pattern for metrics JSON")
    args = parser.parse_args()

    data = defaultdict(dict)
    for path in glob.glob(args.pattern):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        model = m["model"]
        lp = m["lang_pair"]
        data[model][lp] = m["bleu"]

    models = ["smt", "seq2seq", "transformer"]
    header = "| Model            | cs→en | de→en | fr→en | hi→en | ru→en | Avg |"
    sep = "| ---------------- | ----- | ----- | ----- | ----- | ----- | --- |"
    print(header)
    print(sep)
    for model in models:
        row = [model.upper() if model == "smt" else model.capitalize()]
        scores = []
        for lp in LANG_PAIRS:
            s = data.get(model, {}).get(lp, None)
            if s is None:
                row.append(" - ")
            else:
                row.append(f"{s:.2f}")
                scores.append(s)
        avg = sum(scores) / len(scores) if scores else 0.0
        row.append(f"{avg:.2f}")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()

