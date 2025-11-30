import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_PATTERN = re.compile(r"Epoch (\d+) Step (\d+): loss=([0-9.]+)")


def parse_log(path: Path):
    epochs = []
    global_steps = []
    losses = []
    step_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            step = int(m.group(2))
            loss = float(m.group(3))
            step_count += 1
            epochs.append(epoch)
            global_steps.append(step_count)
            losses.append(loss)
    return epochs, global_steps, losses


def plot_loss(log_path: Path, label: str, ax=None):
    if ax is None:
        ax = plt.gca()
    epochs, global_steps, losses = parse_log(log_path)
    if not losses:
        print(f"No loss lines found in {log_path}")
        return ax
    ax.plot(global_steps, losses, label=label, alpha=0.7)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Training Loss")
    return ax


def main():
    base = Path(__file__).resolve().parent.parent
    logs_dir = base / "logs"

    seq2seq_log = logs_dir / "seq2seq_v2.log"
    transformer_log = logs_dir / "transformer_v3.log"

    # 单独绘制 Seq2Seq loss 曲线
    if seq2seq_log.exists():
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        plot_loss(seq2seq_log, label="Seq2Seq", ax=ax1)
        ax1.set_title("Seq2Seq Training Loss (per logged step)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        out_path1 = base / "analysis" / "seq2seq_v2_loss.png"
        fig1.tight_layout()
        fig1.savefig(out_path1, dpi=200)
        print(f"Saved Seq2Seq loss curve to {out_path1}")

    # 单独绘制 Transformer loss 曲线
    if transformer_log.exists():
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        plot_loss(transformer_log, label="Transformer", ax=ax2)
        ax2.set_title("Transformer Training Loss (per logged step)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        out_path2 = base / "analysis" / "transformer_v3_loss.png"
        fig2.tight_layout()
        fig2.savefig(out_path2, dpi=200)
        print(f"Saved Transformer loss curve to {out_path2}")


if __name__ == "__main__":
    main()
