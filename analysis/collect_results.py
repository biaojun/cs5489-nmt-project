"""
Aggregate BLEU / METEOR scores from multilingual JSON results.

Expected JSON:
{
  "model": "seq2seq" | "transformer",
  "lang_pair": "multilingual",
  "split": "test",
  "bleu": ...,
  "meteor": ...
}
"""

import argparse
import glob
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="glob pattern for multilingual metrics JSON files",
    )
    args = parser.parse_args()

    rows = []
    for path in glob.glob(args.pattern):
        with open(path, "r", encoding="utf-8") as f:
            metric = json.load(f)
        rows.append((metric["model"], metric.get("bleu", 0.0), metric.get("meteor", 0.0)))

    header = "| Model        | BLEU | METEOR |"
    sep = "| ------------ | ---- | ------ |"
    print(header)
    print(sep)
    for model, bleu, meteor in sorted(rows):
        name = model.capitalize()
        print(f"| {name:<12} | {bleu:>4.2f} | {meteor:>6.3f} |")


if __name__ == "__main__":
    main()

