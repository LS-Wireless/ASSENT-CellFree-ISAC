# assent_plot_v4.py
import argparse, json, os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


# ------------- helpers -------------
def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def moving_average(x: List[float], k: int) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return np.asarray(x, dtype=float)
    k = min(k, len(x))
    w = np.ones(k, dtype=float) / k
    return np.convolve(np.asarray(x, dtype=float), w, mode="valid")


def get(hist: Dict, key: str, default=None):
    return hist.get(key, default if default is not None else [])


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pad_for_ma(xs: np.ndarray, full_len: int, k: int) -> np.ndarray:
    """Left-pad moving-average result so x-axis aligns with epochs."""
    if k <= 1: return xs
    pad = np.array([np.nan] * (full_len - len(xs)))
    return np.concatenate([pad, xs], axis=0)


# ------------- plotting -------------
def plot_losses(history: Dict, out_dir: str, smooth: int = 1, show_plot: bool = True, save_plot: bool = True):
    ep = np.asarray(get(history, "epoch", []), dtype=int)
    if len(ep) == 0:
        print("No epochs in history; skipping loss plots.")
        return

    # train
    Ltot_t = get(history, "train_loss_total", [])
    Lmulti_t = get(history, "train_loss_multi", [])
    Llin_t = get(history, "train_loss_linear", [])
    Lx_t = get(history, "train_loss_x", [])
    Ltau_t = get(history, "train_loss_tau", [])
    Ls_t = get(history, "train_loss_s", [])
    Lytx_t = get(history, "train_loss_ytx", [])
    Lyrx_t = get(history, "train_loss_yrx", [])

    # val
    Ltot_v = get(history, "val_loss_total", [])
    Lx_v = get(history, "val_loss_x", [])
    Ltau_v = get(history, "val_loss_tau", [])
    Ls_v = get(history, "val_loss_s", [])
    Lytx_v = get(history, "val_loss_ytx", [])
    Lyrx_v = get(history, "val_loss_yrx", [])

    # Apply smoothing (moving average)
    def sm(arr):
        ma = moving_average(arr, smooth)
        return pad_for_ma(ma, len(ep), smooth)

    items = [
        ("loss_total", Ltot_t, Ltot_v, "Total Loss"),
        ("loss_x", Lx_t, Lx_v, "x Loss"),
        ("loss_tau", Ltau_t, Ltau_v, "tau Loss"),
        ("loss_s", Ls_t, Ls_v, "s Loss"),
        ("loss_ytx", Lytx_t, Lytx_v, "y_tx Loss"),
        ("loss_yrx", Lyrx_t, Lyrx_v, "y_rx Loss")
    ]

    for fname, tr, va, title in items:
        if len(tr) == 0 and len(va) == 0:
            continue
        plt.figure(figsize=(7, 4.2))
        if len(tr):
            plt.plot(ep, sm(tr), label="train")
        if len(va):
            plt.plot(ep, sm(va), label="val")
        plt.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        if show_plot: plt.show()

        if save_plot:
            path = os.path.join(out_dir, f"{fname}.png")
            plt.savefig(path, dpi=160)
            plt.close()
            print(f"Saved: {path}")

    plt.figure(figsize=(7, 4.2))
    frame = "train_loss_multi_lin"
    if len(Lmulti_t):
        plt.plot(ep, sm(Lmulti_t), label="loss-multi")
    if len(Llin_t):
        plt.plot(ep, sm(Llin_t), label="loss-linear")
    plt.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    plt.title("Train Loss with Uncertainty Parameters vs Linear")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if show_plot: plt.show()

    if save_plot:
        path = os.path.join(out_dir, f"{frame}.png")
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"Saved: {path}")


def plot_f1(history: Dict, out_dir: str, smooth: int = 1, show_plot: bool = True, save_plot: bool = True):
    ep = np.asarray(get(history, "epoch", []), dtype=int)
    if len(ep) == 0:
        print("No epochs in history; skipping F1 plots.")
        return

    f1x = get(history, "val_f1x", [])
    f1t = get(history, "val_f1tau", [])
    f1s = get(history, "val_f1s", [])
    f1ytx = get(history, "val_f1ytx", [])
    f1yrx = get(history, "val_f1yrx", [])
    thrx = get(history, "thr_x", [])
    thrt = get(history, "thr_tau", [])
    thrs = get(history, "thr_s", [])
    thrytx = get(history, "thr_ytx", [])
    thryrx = get(history, "thr_yrx", [])

    def sm(a):
        return pad_for_ma(moving_average(a, smooth), len(ep), smooth)

    # F1 curves
    plt.figure(figsize=(7, 4.2))
    if len(f1x): plt.plot(ep, sm(f1x), label="x (utility-F1)")
    if len(f1t): plt.plot(ep, sm(f1t), label="tau (macro-F1)")
    if len(f1s): plt.plot(ep, sm(f1s), label="s (macro-F1)")
    if len(f1ytx): plt.plot(ep, sm(f1ytx), label="y_tx (macro-F1)")
    if len(f1yrx): plt.plot(ep, sm(f1yrx), label="y_rx (macro-F1)")
    plt.ylim(0.5, 1.0)
    plt.title("Validation F1 Scores")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if show_plot: plt.show()
    if save_plot:
        path = os.path.join(out_dir, "f1_curves.png")
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"Saved: {path}")

    # Thresholds
    if len(thrx) + len(thrt) + len(thrs) > 0:
        plt.figure(figsize=(7, 4.2))
        if len(thrx): plt.plot(ep, sm(thrx), label="thr_x")
        if len(thrt): plt.plot(ep, sm(thrt), label="thr_tau")
        if len(thrs): plt.plot(ep, sm(thrs), label="thr_s")
        if len(thrytx): plt.plot(ep, sm(thrytx), label="thr_ytx")
        if len(thryrx): plt.plot(ep, sm(thryrx), label="thr_yrx")
        plt.title("Chosen Thresholds (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("Threshold")
        plt.ylim(0.0, 1.0)
        plt.grid(True, ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        if show_plot: plt.show()
        if save_plot:
            path = os.path.join(out_dir, "thresholds.png")
            plt.savefig(path, dpi=160)
            plt.close()
            print(f"Saved: {path}")


def plot_lr(history: Dict, out_dir: str, smooth: int = 1, show_plot: bool = True, save_plot: bool = True):
    ep = np.asarray(get(history, "epoch", []), dtype=int)
    lr = get(history, "lr", [])
    if len(ep) == 0 or len(lr) == 0:
        print("No LR history; skipping LR plot.")
        return
    lr_ma = pad_for_ma(moving_average(lr, smooth), len(ep), smooth)

    plt.figure(figsize=(7, 4.2))
    plt.plot(ep, lr_ma)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    if show_plot: plt.show()
    if save_plot:
        path = os.path.join(out_dir, "lr.png")
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"Saved: {path}")


# ------------- main -------------
if __name__ == "__main__":
    # ap = argparse.ArgumentParser(description="Plot training history for ASSENT model.")
    # ap.add_argument("--history", type=str, required=True, help="Path to history.json")
    # ap.add_argument("--summary", type=str, default=None, help="Optional path to summary.json")
    # ap.add_argument("--out", type=str, default="plots", help="Output directory for figures")
    # ap.add_argument("--smooth", type=int, default=1, help="Moving-average window (epochs). Use 1 to disable.")
    # args = ap.parse_args()

    run_id = 'run_01'
    hist_filename       = '2025-10-20-06-47_history.json'
    summary_filename    = '2025-10-20-06-47_summary.json'

    out_dir = 'plots'
    smooth = 2
    show_plot = True
    save_plot = False

    hist_path = os.path.join('checkpoints', run_id, hist_filename)
    summary_path = os.path.join('checkpoints', run_id, summary_filename)

    if save_plot: ensure_dir(out_dir)
    hist = load_json(hist_path)
    summ = load_json(summary_path)
    # You can annotate plots with best F1s, if desired:
    print(f"Summary: best F1 x={summ.get('best_f1x')}, tau={summ.get('best_f1tau')}, s={summ.get('best_f1s')}")

    plot_losses(hist, out_dir, smooth=smooth, show_plot=show_plot, save_plot=save_plot)
    plot_f1(hist, out_dir, smooth=smooth, show_plot=show_plot, save_plot=save_plot)
    plot_lr(hist, out_dir, smooth=smooth, show_plot=show_plot, save_plot=save_plot)
