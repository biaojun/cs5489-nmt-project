import os

LANG_PAIRS = [
    ("cs", "en"),
    ("de", "en"),
    ("fr", "en"),
    ("hi", "en"),
    ("ru", "en"),
]

LANG_TAGS = {src: f"<{src}>" for src, _ in LANG_PAIRS}


def _merge_split(data_dir: str, split: str, out_dir: str) -> int:
    out_src = os.path.join(out_dir, f"{split}.src")
    out_tgt = os.path.join(out_dir, f"{split}.tgt")
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    with open(out_src, "w", encoding="utf-8") as fsrc, open(
        out_tgt, "w", encoding="utf-8"
    ) as ftgt:
        for src, tgt in LANG_PAIRS:
            pair_dir = os.path.join(data_dir, f"{src}-{tgt}")
            src_path = os.path.join(pair_dir, f"{split}.src")
            tgt_path = os.path.join(pair_dir, f"{split}.tgt")
            if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
                continue
            with open(src_path, "r", encoding="utf-8") as fs, open(
                tgt_path, "r", encoding="utf-8"
            ) as ft:
                for s_line, t_line in zip(fs, ft):
                    s_line = s_line.strip()
                    t_line = t_line.strip()
                    if not s_line or not t_line:
                        continue
                    prefix = LANG_TAGS[src]
                    fsrc.write(f"{prefix} {s_line}\n")
                    ftgt.write(t_line + "\n")
                    total += 1
    return total


def ensure_multilingual_dataset(data_dir: str, out_name: str = "multilingual"):
    """
    Ensure processed_wmt14/multilingual/{split}.src/.tgt exist.
    Builds them on the fly if missing.
    """
    out_dir = os.path.join(data_dir, out_name)
    required = [
        os.path.join(out_dir, f"{split}.src") for split in ["train", "valid", "test"]
    ]
    if all(os.path.exists(path) for path in required):
        return
    os.makedirs(out_dir, exist_ok=True)
    for split in ["train", "valid", "test"]:
        _merge_split(data_dir, split, out_dir)
