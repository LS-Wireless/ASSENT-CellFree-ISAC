
import json
import os
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import src.utils.library as lib


BASE_DIR = os.path.dirname(os.path.abspath(__name__))
if os.path.basename(BASE_DIR) == 'ASSENT-CellFree-ISAC':
    BASE_DIR = os.path.join(BASE_DIR + '/src/learning_v4')
else:
    BASE_DIR = os.path.join(BASE_DIR)
data_dir = os.path.join(BASE_DIR, 'checkpoints')
# ---- Helpers ----
def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


#%% Loading data

# NNConv data
run_id = 'run_03'
hist_filename       = '2025-10-22-05-31_history.json'
summary_filename    = '2025-10-22-05-31_summary.json'
hist_path = os.path.join(data_dir, run_id, hist_filename)
summary_path = os.path.join(data_dir, run_id, summary_filename)

hist_nnconv = load_json(hist_path)
summ_nnconv = load_json(summary_path)

# TransformerConv data
run_id = 'run_02'
hist_filename       = '2025-10-21-02-03_history.json'
summary_filename    = '2025-10-21-02-03_summary.json'
hist_path = os.path.join(data_dir, run_id, hist_filename)
summary_path = os.path.join(data_dir, run_id, summary_filename)

hist_transformer = load_json(hist_path)
summ_transformer = load_json(summary_path)

# GATv2Conv data
run_id = 'run_04'
hist_filename       = '2025-10-31-04-07_history.json'
summary_filename    = '2025-10-31-04-07_summary.json'
hist_path = os.path.join(data_dir, run_id, hist_filename)
summary_path = os.path.join(data_dir, run_id, summary_filename)

hist_gat = load_json(hist_path)
summ_gat = load_json(summary_path)


#%% Plotting (two subplots)


# --- Pub styling (Times NR, vector-friendly) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
})

def _to_array(v, n=None, fill=np.nan):
    if v is None: return np.full(n, fill, dtype=float)
    return np.asarray(v, dtype=float)

def _parse_history_obj(obj):
    """
    Supports either:
      - list of epoch dicts: [{'epoch':..., 'learning_rate':..., 'train_loss_total':..., 'val_loss_total':...}, ...]
      - dict of lists (optionally under 'history')
    Returns: dict with np.arrays: epoch, train_loss_total, val_loss_total, lr (or None if absent)
    """
    if isinstance(obj, list):
        epoch = [r.get("epoch") for r in obj]
        # fallback to 0..N-1 if epochs missing
        if any(e is None for e in epoch):
            epoch = list(range(len(obj)))
        lr   = [r.get("learning_rate", r.get("lr", np.nan)) for r in obj]
        tr   = [r.get("train_loss_total", np.nan) for r in obj]
        val  = [r.get("val_loss_total",   np.nan) for r in obj]
        out = {
            "epoch": _to_array(epoch),
            "train_loss_total": _to_array(tr),
            "val_loss_total":   _to_array(val),
            "lr":               _to_array(lr)
        }
    elif isinstance(obj, dict):
        h = obj.get("history", obj) if isinstance(obj.get("history", None), dict) else obj
        # length fallback from any list we find
        L = None
        for k in ("epoch","train_loss_total","val_loss_total","learning_rate","lr"):
            if isinstance(h.get(k), (list, tuple, np.ndarray)):
                L = len(h[k]); break
        if L is None:
            # try inside nested dicts
            raise ValueError("Unrecognized history format: need list of records or dict of lists.")
        epoch = h.get("epoch", list(range(L)))
        lr    = h.get("learning_rate", h.get("lr", [np.nan]*L))
        tr    = h.get("train_loss_total", [np.nan]*L)
        val   = h.get("val_loss_total",   [np.nan]*L)
        out = {
            "epoch": _to_array(epoch),
            "train_loss_total": _to_array(tr),
            "val_loss_total":   _to_array(val),
            "lr":               _to_array(lr)
        }
    else:
        raise ValueError("history must be a list or dict")
    # if lr is entirely NaN, hide it
    if not np.isfinite(out["lr"]).any():
        out["lr"] = None
    return out

def _ema(x, alpha=0.2):
    x = np.asarray(x, float)
    y = np.copy(x)
    m = np.isfinite(x)
    if not m.any(): return y
    idx = np.where(m)[0]
    y[idx[0]] = x[idx[0]]
    for k in idx[1:]:
        y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
    return y

def plot_train_val_losses_from_histories(hist_nnconv, hist_transformer, hist_gat,
                                         ema_alpha=0.2, show_lr=True,
                                         title_left="Training loss (total)",
                                         title_right="Validation loss (total)"):
    colors = {"NNConv":"tab:blue", "TransformerConv":"tab:orange", "GATv2Conv":"tab:green"}
    runs = {
        "NNConv":          _parse_history_obj(hist_nnconv),
        "TransformerConv": _parse_history_obj(hist_transformer),
        "GATv2Conv":       _parse_history_obj(hist_gat),
    }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axL_lr = axL.twinx() if show_lr else None
    axR_lr = axR.twinx() if show_lr else None
    if show_lr:
        axL_lr.set_zorder(axL.get_zorder() + 1); axL_lr.patch.set_visible(False)
        axR_lr.set_zorder(axR.get_zorder() + 1); axR_lr.patch.set_visible(False)

    linesL, linesR = [], []
    for name, H in runs.items():
        e  = H["epoch"]
        tr = H["train_loss_total"]
        va = H["val_loss_total"]
        lr = H["lr"]

        trp = _ema(tr, ema_alpha) if (ema_alpha and ema_alpha > 0) else tr
        vap = _ema(va, ema_alpha) if (ema_alpha and ema_alpha > 0) else va

        lnL, = axL.plot(e, trp, "-",  lw=2.0, color=colors[name], label=name)
        lnR, = axR.plot(e, vap, "--", lw=2.0, color=colors[name], label=name)
        linesL.append(lnL); linesR.append(lnR)

        if show_lr and lr is not None:
            axL_lr.plot(e, lr, linestyle=(0,(1,1)), lw=1.2, color=colors[name], alpha=0.9)
            axR_lr.plot(e, lr, linestyle=(0,(1,1)), lw=1.2, color=colors[name], alpha=0.9)

    # Labels, grids, legends
    axL.set_xlabel("Epoch"); axL.set_ylabel("Train loss (total)"); axL.set_title(title_left, fontsize=11)
    axR.set_xlabel("Epoch"); axR.set_ylabel("Val loss (total)");   axR.set_title(title_right, fontsize=11)
    for ax in (axL, axR):
        ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.35)
    axL.legend(linesL, [l.get_label() for l in linesL], frameon=False, loc="upper right", title="Train")
    axR.legend(linesR, [l.get_label() for l in linesR], frameon=False, loc="upper right", title="Val")

    if show_lr:
        axL_lr.set_ylabel("Learning rate", color="0.25"); axL_lr.tick_params(axis="y", labelcolor="0.25")
        axR_lr.set_ylabel("Learning rate", color="0.25"); axR_lr.tick_params(axis="y", labelcolor="0.25")
        # Uncomment to show LR in log scale if you used cosine/step schedulers:
        # axL_lr.set_yscale("log"); axR_lr.set_yscale("log")

    plt.show()

# ---- Call it with your already-loaded dicts ----
plot_train_val_losses_from_histories(hist_nnconv, hist_transformer, hist_gat,
                                     ema_alpha=0.1, show_lr=True)


#%% Plotting (one subplot)


# --- Styling (Times NR, vector-friendly) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
})

def _to_array(v, n=None, fill=np.nan):
    if v is None: return np.full(n, fill, dtype=float)
    return np.asarray(v, dtype=float)

def _parse_history_obj(obj):
    """
    Accepts:
      - list of records: [{'epoch':..., 'learning_rate'/'lr':..., 'train_loss_total':..., 'val_loss_total':...}, ...]
      - dict of lists (optionally under 'history')
    Returns arrays: epoch, train_loss_total, val_loss_total, lr (or None if absent)
    """
    if isinstance(obj, list):
        epoch = [r.get("epoch") for r in obj]
        if any(e is None for e in epoch):
            epoch = list(range(len(obj)))
        lr   = [r.get("learning_rate", r.get("lr", np.nan)) for r in obj]
        tr   = [r.get("train_loss_total", np.nan) for r in obj]
        val  = [r.get("val_loss_total",   np.nan) for r in obj]
        out = {"epoch": _to_array(epoch), "train_loss_total": _to_array(tr),
               "val_loss_total": _to_array(val), "lr": _to_array(lr)}
    elif isinstance(obj, dict):
        h = obj.get("history", obj) if isinstance(obj.get("history", None), dict) else obj
        L = None
        for k in ("epoch","train_loss_total","val_loss_total","learning_rate","lr"):
            if isinstance(h.get(k), (list, tuple, np.ndarray)):
                L = len(h[k]); break
        if L is None:
            raise ValueError("Unrecognized history format.")
        epoch = h.get("epoch", list(range(L)))
        lr    = h.get("learning_rate", h.get("lr", [np.nan]*L))
        tr    = h.get("train_loss_total", [np.nan]*L)
        val   = h.get("val_loss_total",   [np.nan]*L)
        out = {"epoch": _to_array(epoch), "train_loss_total": _to_array(tr),
               "val_loss_total": _to_array(val), "lr": _to_array(lr)}
    else:
        raise ValueError("history must be a list or dict")
    if not np.isfinite(out["lr"]).any():
        out["lr"] = None
    return out

def _ema(x, alpha=0.2):
    x = np.asarray(x, float); y = np.copy(x)
    m = np.isfinite(x)
    if not m.any(): return y
    idx = np.where(m)[0]
    y[idx[0]] = x[idx[0]]
    for k in idx[1:]:
        y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
    return y


#%% Figure 1

ema_alpha = 0.1
show_lr = True
lr_log = False

lw = 1.8
lw_lr = 1.4
colors = {
    "NNConv":          "#4477AA",  # blue
    "TransformerConv": "#EE7733",  # orange
    "GATv2Conv":       "#009988",  # teal
    "lr": "tab:purple"
}
runs = {
    "NNConv":          _parse_history_obj(hist_nnconv),
    "TransformerConv": _parse_history_obj(hist_transformer),
    "GATv2Conv":       _parse_history_obj(hist_gat),
}

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax_lr = ax.twinx() if show_lr else None
if show_lr:
    ax_lr.set_zorder(ax.get_zorder() + 1)
    ax_lr.patch.set_visible(False)
    if lr_log:
        ax_lr.set_yscale("log")

# Plot train (solid) & val (dashed) per architecture
from matplotlib.lines import Line2D
arch_handles = []
for name, H in runs.items():
    e  = H["epoch"]
    tr = H["train_loss_total"]; va = H["val_loss_total"]; lr = H["lr"]
    trp = _ema(tr, ema_alpha) if (ema_alpha and ema_alpha > 0) else tr
    vap = _ema(va, ema_alpha) if (ema_alpha and ema_alpha > 0) else va

    ax.plot(e, trp, "-",  lw=lw, color=colors[name], label=f"{name} – Train")
    ax.plot(e, vap, "--", lw=lw, color=colors[name], label=f"{name} – Val")

    if show_lr and lr is not None:
        ax_lr.plot(e, lr, linestyle=(0,(1,1)), lw=lw_lr, color=colors['lr'], alpha=0.9)

    # For a clean architecture legend (color only)
    arch_handles.append(Line2D([0],[0], color=colors[name], lw=2.0, label=name))

# Axes labels
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (total)")
if show_lr:
    ax_lr.set_ylabel("Learning rate", color=colors['lr'])
    ax_lr.tick_params(axis="y", labelcolor=colors['lr'])
    ax_lr.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_lr.set_ylim(bottom=0)

ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.35)

# Legends: (1) architectures (colors), (2) line styles (train vs val), (3) optional LR style
style_handles = [
    Line2D([0],[0], color="0.2", lw=2.0, linestyle="-",  label="Train loss"),
    Line2D([0],[0], color="0.2", lw=2.0, linestyle="--", label="Val loss"),
]
leg1 = ax.legend(handles=arch_handles, frameon=True, loc="upper right", title="Architecture")
ax.add_artist(leg1)
leg2 = ax.legend(handles=style_handles, frameon=True, loc="lower left")
if show_lr:
    lr_handle = [Line2D([0],[0], color=colors['lr'], lw=lw_lr, linestyle=(0,(1,1)), label="Learning rate (right)")]
    ax.legend(handles=lr_handle, frameon=True, loc="lower left")

plt.show()


#%% ============= NNConv - Load model state for inference on test set

from src.learning_v4.assent_data_utils_v4 import *
from src.learning_v4.assent_data_v4 import GraphDataset
from torch_geometric.loader import DataLoader

nsamps = 10000
results = data_loader(nsamps=nsamps, CONSOLE_RUN=True)

run_id = 'run_03'
filename = 'metadata.json'
metadata_path = os.path.join(data_dir, run_id, filename)
metadata_nnconv = load_json(metadata_path)
lib.print_log(tag='LOAD', message=f"Loaded NNConv metadata from {metadata_path}")

batch_size = metadata_nnconv['TrainConfig']['batch_size']
device = metadata_nnconv['TrainConfig']['device']

# Prepare dataset
idx = np.random.RandomState(metadata_nnconv['config']['split_seed']).permutation(nsamps)
ntr = int(0.8 * nsamps)
nva = int(0.1 * nsamps)
tr_idx, va_idx = idx[:ntr], idx[ntr:ntr + nva]
te_idx = idx[ntr + nva:]

ds = GraphDataset(results["G_comm"], results["S_comm"], results["G_sens"],
                  results["alpha"], results["lambda_cu"], results["lambda_tg"],
                  results["solution"], N_RF=results["N_RF"], device=device,
                  K_tx_milp=results["K_tx_milp"], K_rx_milp=results["K_rx_milp"])

from torch.utils.data import Subset
tr_set, val_set, test_set = Subset(ds, tr_idx.tolist()), Subset(ds, va_idx.tolist()), Subset(ds, te_idx.tolist())

lib.print_log(tag='RUN',
              message=f"Training set size: {len(tr_set)}; Validation set size: {len(val_set)}; Test set size: {len(test_set)}")
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

nnconv_pareto_id = 0
thr_dict_nnconv = {
    "thr_x": summ_nnconv['pareto'][nnconv_pareto_id]['thr_x'],
    "thr_tau": summ_nnconv['pareto'][nnconv_pareto_id]['thr_tau'],
    "thr_s": summ_nnconv['pareto'][nnconv_pareto_id]['thr_s'],
    "thr_ytx": summ_nnconv['pareto'][nnconv_pareto_id]['thr_ytx'],
    "thr_yrx": summ_nnconv['pareto'][nnconv_pareto_id]['thr_yrx'],
}


#%% NNConv - Load model state

from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer

model_nnconv = ASSENTGNN(hidden=metadata_nnconv['config']['nhidden'], layers=metadata_nnconv['config']['nlayers'], drop=metadata_nnconv['config']['drop']).to(device)


model_path = os.path.join(data_dir, run_id, "pareto/pareto_ep041.pt")
state = torch.load(model_path, map_location=device)
model_nnconv.load_state_dict(state)
model_nnconv.eval()
lib.print_log(tag='LOAD', message=f"Loaded NNConv model from {model_path}")


#%% NNConv - Inference on test set: NNConv

from src.learning_v4.assent_test_utils_v4 import run_inference

probs_cat_nnconv, preds_bin_nnconv, y_cat_nnconv, metrics_nnconv = run_inference(model=model_nnconv, loader=test_loader, device=device, thr_dict=thr_dict_nnconv)
lib.print_log(tag='RUN', message=f"Finished inference on test set for NNConv model.")


#%% NNConv - Print summary metrics
from contextlib import redirect_stdout

from src.learning_v4.assent_test_utils_v4 import summarize_inference, print_metrics_table

TASKS = ["x", "tau", "s", "ytx", "yrx"]
metrics_dict_nnconv = summarize_inference(probs_cat_nnconv, preds_bin_nnconv, y_cat_nnconv, keys=TASKS)
txt_filename = os.path.join(data_dir, run_id, "metrics_summary_nnconv.txt")
with open(txt_filename, "w", encoding="utf-8") as f, redirect_stdout(f):
    print_metrics_table(metrics_dict_nnconv)
    lib.print_log(tag='PRINT', message=f"Summary metrics for NNConv model.")


#%% NNConv - Compute MILP vs ASSENT objective value on test set: NNConv

from src.learning_v4.assent_test_utils_v4 import milp_vs_assent

recs_nnconv = milp_vs_assent(model=model_nnconv, test_loader=test_loader, test_idx=te_idx, device=device, results=results, thr_dict=thr_dict_nnconv)
lib.print_log(tag='RUN', message=f"Finished computing MILP vs ASSENT objective value on test set for NNConv model.")

#%% NNConv - Plotting MILP vs ASSENT utility

Ugt_nnconv = np.array([r['U_gt'] for r in recs_nnconv])
Upd_nnconv = np.array([r['U_pd'] for r in recs_nnconv])
plt.figure()
plt.scatter(Ugt_nnconv, Upd_nnconv, s=12, alpha=0.5)
lims = [min(Ugt_nnconv.min(), Upd_nnconv.min()), max(Ugt_nnconv.max(), Upd_nnconv.max())]
plt.plot(lims, lims, '--', alpha=0.5, color='tab:red')
plt.xlabel('MILP Utility (ground truth)')
plt.ylabel('ASSENT Utility (predicted)')
plt.title('Utility Comparison on Test Set')
plt.tight_layout()
plt.show()


#%% ============= TransformerConv - Load model state for inference on test set


run_id = 'run_02'
filename = 'metadata.json'
metadata_path = os.path.join(data_dir, run_id, filename)
metadata_transformer = load_json(metadata_path)
lib.print_log(tag='LOAD', message=f"Loaded TransformerConv metadata from {metadata_path}")

batch_size = metadata_transformer['TrainConfig']['batch_size']
device = metadata_transformer['TrainConfig']['device']

# Prepare dataset
idx = np.random.RandomState(metadata_transformer['config']['split_seed']).permutation(nsamps)
ntr = int(0.8 * nsamps)
nva = int(0.1 * nsamps)
tr_idx, va_idx = idx[:ntr], idx[ntr:ntr + nva]
test_idx_transformer = idx[ntr + nva:]

ds = GraphDataset(results["G_comm"], results["S_comm"], results["G_sens"],
                  results["alpha"], results["lambda_cu"], results["lambda_tg"],
                  results["solution"], N_RF=results["N_RF"], device=device,
                  K_tx_milp=results["K_tx_milp"], K_rx_milp=results["K_rx_milp"])

from torch.utils.data import Subset
tr_set, val_set, test_set = Subset(ds, tr_idx.tolist()), Subset(ds, va_idx.tolist()), Subset(ds, test_idx_transformer.tolist())

lib.print_log(tag='RUN',
              message=f"Training set size: {len(tr_set)}; Validation set size: {len(val_set)}; Test set size: {len(test_set)}")
test_loader_transformer = DataLoader(test_set, batch_size=batch_size, shuffle=False)

transformer_pareto_id = 0
thr_dict_transformer = {
    "thr_x": summ_nnconv['pareto'][transformer_pareto_id]['thr_x'],
    "thr_tau": summ_nnconv['pareto'][transformer_pareto_id]['thr_tau'],
    "thr_s": summ_nnconv['pareto'][transformer_pareto_id]['thr_s'],
    "thr_ytx": summ_nnconv['pareto'][transformer_pareto_id]['thr_ytx'],
    "thr_yrx": summ_nnconv['pareto'][transformer_pareto_id]['thr_yrx'],
}


#%% TransformerConv - Load model state


model_transformer = ASSENTGNN_Transformer(hidden=metadata_transformer['config']['nhidden'], layers=metadata_transformer['config']['nlayers'], drop=metadata_transformer['config']['drop'],
                                          heads=metadata_transformer['config']['transformer_heads'], beta=metadata_transformer['config']['transformer_beta']).to(device)


model_path = os.path.join(data_dir, run_id, "pareto/pareto_ep077.pt")
state = torch.load(model_path, map_location=device)
model_transformer.load_state_dict(state)
model_transformer.eval()
lib.print_log(tag='LOAD', message=f"Loaded TransformerConv model from {model_path}")


#%% TransformerConv - Inference on test set: TransformerConv

from src.learning_v4.assent_test_utils_v4 import run_inference

probs_cat_transformer, preds_bin_transformer, y_cat_transformer, metrics_transformer = run_inference(model=model_transformer, loader=test_loader_transformer,
                                                                                                     device=device, thr_dict=thr_dict_transformer)
lib.print_log(tag='RUN', message=f"Finished inference on test set for TransformerConv model.")


#%% TransformerConv - Print summary metrics
from contextlib import redirect_stdout

from src.learning_v4.assent_test_utils_v4 import summarize_inference, print_metrics_table

TASKS = ["x", "tau", "s", "ytx", "yrx"]
metrics_dict_transformer = summarize_inference(probs_cat_transformer, preds_bin_transformer, y_cat_transformer, keys=TASKS)
txt_filename = os.path.join(data_dir, run_id, "metrics_summary_transformer.txt")
with open(txt_filename, "w", encoding="utf-8") as f, redirect_stdout(f):
    print_metrics_table(metrics_dict_transformer)
    lib.print_log(tag='PRINT', message=f"Summary metrics for TransformerConv model.")


#%% TransformerConv - Compute MILP vs ASSENT objective value on test set: TransformerConv

from src.learning_v4.assent_test_utils_v4 import milp_vs_assent

recs_transformer = milp_vs_assent(model=model_transformer, test_loader=test_loader_transformer, test_idx=test_idx_transformer, device=device,
                                  results=results, thr_dict=thr_dict_transformer)
lib.print_log(tag='RUN', message=f"Finished computing MILP vs ASSENT objective value on test set for TransformerConv model.")

#%% TransformerConv - Plotting MILP vs ASSENT utility

Ugt_transformer = np.array([r['U_gt'] for r in recs_transformer])
Upd_transformer = np.array([r['U_pd'] for r in recs_transformer])

plt.figure()
plt.scatter(Ugt_transformer, Upd_transformer, s=12, alpha=0.5)
lims = [min(Ugt_transformer.min(), Upd_transformer.min()), max(Ugt_transformer.max(), Upd_transformer.max())]
plt.plot(lims, lims, '--', alpha=0.5, color='tab:red')
plt.xlabel('MILP Utility (ground truth)')
plt.ylabel('ASSENT Utility (predicted)')
plt.title('Utility Comparison on Test Set')
plt.tight_layout()
plt.show()


#%% ============= GATv2Conv - Load model state for inference on test set


run_id = 'run_04'
filename = 'metadata.json'
metadata_path = os.path.join(data_dir, run_id, filename)
metadata_gat = load_json(metadata_path)
lib.print_log(tag='LOAD', message=f"Loaded GATv2Conv metadata from {metadata_path}")

batch_size = metadata_gat['TrainConfig']['batch_size']
device = metadata_gat['TrainConfig']['device']

# Prepare dataset
idx = np.random.RandomState(metadata_gat['config']['split_seed']).permutation(nsamps)
ntr = int(0.8 * nsamps)
nva = int(0.1 * nsamps)
tr_idx, va_idx = idx[:ntr], idx[ntr:ntr + nva]
test_idx_gat = idx[ntr + nva:]

ds = GraphDataset(results["G_comm"], results["S_comm"], results["G_sens"],
                  results["alpha"], results["lambda_cu"], results["lambda_tg"],
                  results["solution"], N_RF=results["N_RF"], device=device,
                  K_tx_milp=results["K_tx_milp"], K_rx_milp=results["K_rx_milp"])

from torch.utils.data import Subset
tr_set, val_set, test_set = Subset(ds, tr_idx.tolist()), Subset(ds, va_idx.tolist()), Subset(ds, test_idx_gat.tolist())

lib.print_log(tag='RUN',
              message=f"Training set size: {len(tr_set)}; Validation set size: {len(val_set)}; Test set size: {len(test_set)}")
test_loader_gat = DataLoader(test_set, batch_size=batch_size, shuffle=False)

gat_pareto_id = 0
thr_dict_gat = {
    "thr_x": summ_nnconv['pareto'][gat_pareto_id]['thr_x'],
    "thr_tau": summ_nnconv['pareto'][gat_pareto_id]['thr_tau'],
    "thr_s": summ_nnconv['pareto'][gat_pareto_id]['thr_s'],
    "thr_ytx": summ_nnconv['pareto'][gat_pareto_id]['thr_ytx'],
    "thr_yrx": summ_nnconv['pareto'][gat_pareto_id]['thr_yrx'],
}


#%% GATv2Conv - Load model state


model_gat = ASSENTGNN_GAT(hidden=metadata_gat['config']['nhidden'], layers=metadata_gat['config']['nlayers'], drop=metadata_gat['config']['drop'],
                          heads=metadata_gat['config']['gatv2_heads']).to(device)


model_path = os.path.join(data_dir, run_id, "pareto/pareto_ep092.pt")
state = torch.load(model_path, map_location=device)
model_gat.load_state_dict(state)
model_gat.eval()
lib.print_log(tag='LOAD', message=f"Loaded GATv2Conv model from {model_path}")

#%% GATv2Conv - Inference on test set: GATv2Conv

from src.learning_v4.assent_test_utils_v4 import run_inference

probs_cat_gat, preds_bin_gat, y_cat_gat, metrics_gat = run_inference(model=model_gat, loader=test_loader_gat, device=device, thr_dict=thr_dict_gat)
lib.print_log(tag='RUN', message=f"Finished inference on test set for GATv2Conv model.")


#%% GATv2Conv - Print summary metrics
from contextlib import redirect_stdout

from src.learning_v4.assent_test_utils_v4 import summarize_inference, print_metrics_table

TASKS = ["x", "tau", "s", "ytx", "yrx"]
metrics_dict_gat = summarize_inference(probs_cat_gat, preds_bin_gat, y_cat_gat, keys=TASKS)
txt_filename = os.path.join(data_dir, run_id, "metrics_summary_gat.txt")
with open(txt_filename, "w", encoding="utf-8") as f, redirect_stdout(f):
    print_metrics_table(metrics_dict_gat)
    lib.print_log(tag='PRINT', message=f"Summary metrics for GATv2Conv model.")


#%% GATv2Conv - Compute MILP vs ASSENT objective value on test set: GATv2Conv

from src.learning_v4.assent_test_utils_v4 import milp_vs_assent

recs_gat = milp_vs_assent(model=model_gat, test_loader=test_loader_gat, test_idx=test_idx_gat, device=device,
                          results=results, thr_dict=thr_dict_gat)
lib.print_log(tag='RUN', message=f"Finished computing MILP vs ASSENT objective value on test set for GATv2Conv model.")

#%% GATv2Conv - Plotting MILP vs ASSENT utility

Ugt_gat = np.array([r['U_gt'] for r in recs_gat])
Upd_gat = np.array([r['U_pd'] for r in recs_gat])

plt.figure()
plt.scatter(Ugt_gat, Upd_gat, s=12, alpha=0.5)
lims = [min(Ugt_gat.min(), Upd_gat.min()), max(Ugt_gat.max(), Upd_gat.max())]
plt.plot(lims, lims, '--', alpha=0.5, color='tab:red')
plt.xlabel('MILP Utility (ground truth)')
plt.ylabel('ASSENT Utility (predicted)')
plt.title('Utility Comparison on Test Set')
plt.tight_layout()
plt.show()


#%% Objective value bar plot

Ugt_milp = np.mean(np.array([np.mean(Ugt_nnconv), np.mean(Ugt_transformer), np.mean(Ugt_gat)]))
labels = ["MILP (GT)", "NNConv", "TransformerConv", "GATv2Conv"]
means  = [np.mean(Ugt_gat), np.mean(Upd_nnconv), np.mean(Upd_transformer), np.mean(Upd_gat)]      # normalized to MILP mean
stds   = [np.std(Ugt_nnconv), np.std(Upd_nnconv), np.std(Upd_transformer), np.std(Upd_gat)]

colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:purple']

lw = 1.5
font_size = 11
text_font_size = 12
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,          # base font size
    'axes.labelsize': font_size,     # x and y labels
    'xtick.labelsize': font_size,    # x ticks
    'ytick.labelsize': font_size,    # y ticks
    'legend.fontsize': 11,    # legend
    'axes.titlesize': 10,    # title
})

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=lw)

ax.set_ylabel("Total Objective Value")
ax.grid(axis="y", linestyle="--", alpha=0.5)

fmt="{:.3f}"
for rect, m in zip(bars, means):
    x_center = rect.get_x() + rect.get_width() / 2.0
    y_text = 2.3
    ax.text(
        x_center, y_text, fmt.format(m),
        ha="center", va="center",
        fontsize=text_font_size,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", edgecolor="black", alpha=1.0)
    )

fig.tight_layout()
plt.show()

#%% Figure 2 of the paper

SAVE_FIG = True
save_path = os.path.join(BASE_DIR, 'figures')
os.makedirs(save_path, exist_ok=True)

font_size = 12
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,          # base font size
    'axes.labelsize': font_size,     # x and y labels
    'xtick.labelsize': font_size,    # x ticks
    'ytick.labelsize': font_size,    # y ticks
    'legend.fontsize': 11,    # legend
    'axes.titlesize': 10,    # title
})






fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)

ema_alpha = 0.1
show_lr = True
lr_log = False

lw = 2
lw_lr = 1.5
colors = {
    "NNConv":          "tab:blue",  # blue
    "TransformerConv": "tab:red",  # orange
    "GATv2Conv":       "tab:purple",  # teal
    "lr": "tab:green",
    "lr_dark": "#1e7d1e"
}
runs = {
    "NNConv":          _parse_history_obj(hist_nnconv),
    "TransformerConv": _parse_history_obj(hist_transformer),
    "GATv2Conv":       _parse_history_obj(hist_gat),
}

ax_lr = ax1.twinx() if show_lr else None
if show_lr:
    ax_lr.set_zorder(ax.get_zorder() + 1)
    ax_lr.patch.set_visible(False)
    if lr_log:
        ax_lr.set_yscale("log")

# Plot train (solid) & val (dashed) per architecture
from matplotlib.lines import Line2D
arch_handles = []
for name, H in runs.items():
    e  = H["epoch"]
    tr = H["train_loss_total"]; va = H["val_loss_total"]; lr = H["lr"]
    trp = _ema(tr, ema_alpha) if (ema_alpha and ema_alpha > 0) else tr
    vap = _ema(va, ema_alpha) if (ema_alpha and ema_alpha > 0) else va

    ax1.plot(e, trp, "-", marker='none', markevery=10, markerfacecolor='none', lw=lw, color=colors[name], label=f"{name} – Train")
    ax1.plot(e, vap, "--", marker='none', markevery=10, markerfacecolor='none', lw=lw, color=colors[name], label=f"{name} – Val")

    if show_lr and lr is not None:
        ax_lr.plot(e, lr, linestyle=(0,(1,1)), lw=lw_lr, color=colors['lr'], alpha=0.9)

    # For a clean architecture legend (color only)
    arch_handles.append(Line2D([0],[0], color=colors[name], lw=2.0, label=name))

# Axes labels
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (total)")
if show_lr:
    ax_lr.set_ylabel("Learning rate", color=colors['lr_dark'])
    ax_lr.tick_params(axis="y", labelcolor=colors['lr_dark'])
    ax_lr.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_lr.set_ylim(bottom=0)

ax1.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.35)

# Legends: (1) architectures (colors), (2) line styles (train vs val), (3) optional LR style
style_handles = [
    Line2D([0],[0], color="0.2", lw=2.0, linestyle="-",  label="Train loss"),
    Line2D([0],[0], color="0.2", lw=2.0, linestyle="--", label="Val loss"),
]
leg1 = ax1.legend(handles=arch_handles, frameon=True, loc="upper right", title="Architecture")
ax1.add_artist(leg1)
leg2 = ax1.legend(handles=style_handles, frameon=True, loc="lower left")
if show_lr:
    lr_handle = [Line2D([0],[0], color=colors['lr'], lw=lw_lr, linestyle=(0,(1,1)), label="Learning rate (right)")]
    ax1.legend(handles=lr_handle, frameon=True, loc="lower left")
ax1.set_xlim(0, 85)


plt.rcParams.update({
    'font.size': 12,          # base font size
    'axes.labelsize': 11,     # x and y labels
    'xtick.labelsize': 11,    # x ticks
})

ax2 = fig.add_subplot(1, 2, 2)


Ugt_milp = np.mean(np.array([np.mean(Ugt_nnconv), np.mean(Ugt_transformer), np.mean(Ugt_gat)]))
labels = ["MILP (GT)", "NNConv", "TransformerConv", "GATv2Conv"]
means  = [np.mean(Ugt_gat), np.mean(Upd_nnconv), np.mean(Upd_transformer), np.mean(Upd_gat)]      # normalized to MILP mean
stds   = [np.std(Ugt_nnconv), np.std(Upd_nnconv), np.std(Upd_transformer), np.std(Upd_gat)]

colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:purple']

lw = 1.5
text_font_size = 12

bars = ax2.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=lw)

ax2.set_ylabel("Total Objective Value")
ax2.grid(axis="y", linestyle="--", alpha=0.5)

fmt = "{:.3f}"
for rect, m in zip(bars, means):
    x_center = rect.get_x() + rect.get_width() / 2.0
    y_text = 2.3
    ax2.text(
        x_center, y_text, fmt.format(m),
        ha="center", va="center",
        fontsize=text_font_size,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", edgecolor="black", alpha=1.0)
    )

fig.tight_layout(pad=0.8)
plt.show()

# Save a camera-ready PDF/PNG
if SAVE_FIG:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = os.path.join(save_path, timestamp + "_loss_and_objVal.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
    filename = os.path.join(save_path, timestamp + f"_loss_and_objVal.pdf")
    fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
    lib.print_log(tag='SAVE', message=f"Saved figures to '{save_path}'")
