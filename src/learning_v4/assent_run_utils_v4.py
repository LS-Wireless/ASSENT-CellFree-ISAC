# assent_run_utils_v4.py

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


# -------------------------------------------------------------------
# TrainConfig
# -------------------------------------------------------------------
@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    eps_smooth: float = 0.05  # label smoothing ε for BCE
    rf_cap: float = 4.0  # N_RF (scalar cap per AP)
    # Loss weights
    w_Lrf: float = 0.2  # weight for RF penalty
    w_Lcoup: float = 0.2  # weight for x–tau coupling penalty
    w_Ltau: float = 1.0  # weight for tau BCE
    w_Ls: float = 1.0  # weight for s BCE

    gain_col_x: int = 0  # which edge_attr column is G_comm (normalized)
    thresh_min: float = 0.2
    thresh_max: float = 0.8
    thresh_steps: int = 31
    grad_clip: float = 1.0
    amp: bool = False  # mixed precision toggle
    # LR scheduler
    warmup_epochs: int = 5  # LR scheduler warmup epochs
    lr_min: float = 1e-5  # min LR for scheduler
    # early stopping
    es_metric: str = "val_loss_total"  # or: "f1_combo", "val_loss_total_raw"
    es_mode: str = "min"  # "min" for loss, "max" for F1
    es_patience: int = 12  # epochs without improvement before stop
    es_min_delta: float = 1e-4  # required improvement margin
    # AP-Target edge y_tx/y_rx
    w_Lytx: float = 1.0  # weight for y_tx BCE
    w_Lyrx: float = 1.0  # weight for y_rx BCE
    w_cap: float = 0.5  # weight for per-target cap penalties
    w_tau_match: float = 0.2  # weight for y–tau consistency
    w_s_gate: float = 0.2  # weight for target gating on y
    K_tx_milp: int = 2  # cap per target for y_tx
    K_rx_milp: int = 2  # cap per target for y_rx
    # Pareto setting
    pareto_max_keep: int = 5


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
@torch.no_grad()
def compute_pos_weight_x(loader, device):
    pos = neg = 0
    for b in loader:
        y = b[('ap', 'serves', 'user')].y.view(-1)
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    w = float(neg / max(pos, 1))
    return torch.tensor(w, device=device, dtype=torch.float32)


@torch.no_grad()
def class_weights_node(loader, ntype: str, device):
    pos = neg = 0
    for b in loader:
        if ntype not in b.node_types or 'y' not in b[ntype]:
            continue
        y = b[ntype].y.view(-1)
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    # weights to apply per-sample: w1 to y=1, w0 to y=0
    w1 = neg / max(pos, 1)
    w0 = pos / max(neg, 1)
    return torch.tensor(w0, device=device), torch.tensor(w1, device=device)


def edge_util_weights(batch, gain_col: int = 0):
    """
    w_{a,u} = (1-alpha) * lambda_cu[u] * Gnorm[a,u]
    Normalized by mean to keep loss scale stable.
    """
    alpha = batch['ap'].x[0, 0]  # scalar per-graph
    ei = batch[('ap', 'serves', 'user')].edge_index
    u_idx = ei[1]
    lam_u = batch['user'].x[:, 1]  # [U]
    Gnorm = batch[('ap', 'serves', 'user')].edge_attr[:, gain_col]  # [E]
    w = (1.0 - alpha) * lam_u[u_idx] * Gnorm
    return (w / (w.mean() + 1e-8)).clamp_(0.25, 4.0)


def rf_penalty_from_logits(batch, x_logits, rf_cap: float):
    probs = torch.sigmoid(x_logits.view(-1))  # [E]
    ei = batch[('ap', 'serves', 'user')].edge_index
    ap = ei[0].long()
    A = batch['ap'].x.size(0)
    per_ap = torch.zeros(A, device=probs.device, dtype=probs.dtype).scatter_add_(0, ap, probs)
    return (per_ap - float(rf_cap)).clamp(min=0).mean()


def tau_coupling_penalty(batch, x_logits, tau_logits):
    ei = batch[('ap', 'serves', 'user')].edge_index
    ap = ei[0].long()
    xprob = torch.sigmoid(x_logits.view(-1))
    tauprob = torch.sigmoid(tau_logits.view(-1))
    return (xprob - tauprob[ap]).clamp(min=0).mean()


# ------------------------- Utilities for AP-Target edges y_tx/y_rx --------------------------------
@torch.no_grad()
def compute_pos_weight_edge(loader, rel, device):
    pos = neg = 0
    for b in loader:
        if 'y' not in b[rel]: continue
        y = b[rel].y.view(-1)
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    return torch.tensor(neg / max(pos, 1), dtype=torch.float32, device=device)


def cap_penalty(batch, edge_logits, rel, K_cap, s_ref):
    # sum over APs for each target
    ei = batch[rel].edge_index  # [2,E]
    tgt = ei[1].long()  # target index per edge
    probs = torch.sigmoid(edge_logits.view(-1))
    T = batch['target'].x.size(0)
    per_t_sum = torch.zeros(T, device=probs.device, dtype=probs.dtype).scatter_add_(0, tgt, probs)
    # hinge: max( sum - K*s_ref , 0 )
    return (per_t_sum - K_cap * s_ref).clamp(min=0).mean()


def get_s_ref(batch, s_logit):
    if 'y' in batch['target']:
        return batch['target'].y.view(-1)  # GT during train/val
    else:
        return torch.sigmoid(s_logit.view(-1)).detach()  # fallback


def tau_match_penalty(batch, ytx_logits, yrx_logits, tau_logits):
    ei_tx = batch[('ap', 'senses_tx', 'target')].edge_index
    ap_tx = ei_tx[0].long()
    ei_rx = batch[('ap', 'senses_rx', 'target')].edge_index
    ap_rx = ei_rx[0].long()

    p_tau = torch.sigmoid(tau_logits.view(-1))  # [A]
    p_tx = torch.sigmoid(ytx_logits.view(-1))  # [E_tx]
    p_rx = torch.sigmoid(yrx_logits.view(-1))  # [E_rx]

    pen_rx = (p_rx * p_tau[ap_rx]).mean()  # encourage tau low where y_rx high
    pen_tx = (p_tx * (1 - p_tau[ap_tx])).mean()  # encourage tau high where y_tx high
    return pen_tx, pen_rx


def s_gate_penalty(batch, y_logits, rel, s_ref):
    ei = batch[rel].edge_index
    tgt = ei[1].long()
    p = torch.sigmoid(y_logits.view(-1))
    return (p * (1 - s_ref[tgt])).mean()


# ------------------------- Thresholding --------------------------------
@torch.no_grad()
def best_threshold_utility_f1_x(val_loader, model, device, gain_col=0, t_min=0.2, t_max=0.8, steps=31):
    model.eval()
    ps, ys, ws = [], [], []
    for b in val_loader:
        b = b.to(device)
        out = model(b)
        logits = out['x_logit'] if isinstance(out, dict) else out
        ps.append(torch.sigmoid(logits).view(-1).cpu())
        ys.append(b[('ap', 'serves', 'user')].y.view(-1).cpu())
        ws.append(edge_util_weights(b, gain_col).view(-1).cpu())
    if not ps:
        return 0.5, 0.0, (0.0, 0.0)
    p = torch.cat(ps).numpy()
    y = torch.cat(ys).numpy().astype(np.int32)
    w = torch.cat(ws).numpy().astype(np.float32)

    best_t, best_f1, best_pr = 0.5, 0.0, (0.0, 0.0)
    for t in np.linspace(t_min, t_max, steps):
        yhat = (p > t).astype(np.int32)
        tp_u = w[(yhat == 1) & (y == 1)].sum()
        fp_u = w[(yhat == 1) & (y == 0)].sum()
        fn_u = w[(yhat == 0) & (y == 1)].sum()
        Pu = tp_u / (tp_u + fp_u + 1e-9)
        Ru = tp_u / (tp_u + fn_u + 1e-9)
        f1u = (2 * Pu * Ru) / (Pu + Ru + 1e-9)
        if f1u > best_f1:
            best_f1, best_t, best_pr = f1u, t, (Pu, Ru)
    return float(best_t), float(best_f1), (float(best_pr[0]), float(best_pr[1]))


@torch.no_grad()
def best_threshold_macro_f1_nodes(val_loader, model, device, ntype: str, t_min=0.1, t_max=0.9, steps=33):
    from sklearn.metrics import f1_score
    model.eval()
    ps, ys = [], []
    for b in val_loader:
        b = b.to(device)
        out = model(b)
        if ntype == 'ap':
            logits = out['tau_logit']
        elif ntype == 'target':
            logits = out['s_logit']
        else:
            continue
        ps.append(torch.sigmoid(logits).view(-1).cpu())
        if 'y' in b[ntype]:
            ys.append(b[ntype].y.view(-1).cpu())
    if not ps or not ys:
        return 0.5, 0.0
    p = torch.cat(ps).numpy()
    y = torch.cat(ys).numpy().astype(np.int32)

    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(t_min, t_max, steps):
        yhat = (p > t).astype(np.int32)
        f1m = f1_score(y, yhat, average='macro')
        if f1m > best_f1:
            best_f1, best_t = f1m, t
    return float(best_t), float(best_f1)


@torch.no_grad()
def best_threshold_macro_f1_edge(val_loader, model, device, rel, t_min=0.1, t_max=0.9, steps=33):
    from sklearn.metrics import f1_score
    model.eval()
    ps = []
    ys = []
    for b in val_loader:
        b = b.to(device)
        out = model(b)
        if rel == ('ap', 'senses_tx', 'target'):
            logits = out['ytx_logit']
        else:
            logits = out['yrx_logit']
        if 'y' not in b[rel]: continue
        ps.append(torch.sigmoid(logits).view(-1).cpu())
        ys.append(b[rel].y.view(-1).cpu())
    if not ps: return 0.5, 0.0
    p = torch.cat(ps).numpy()
    y = torch.cat(ys).numpy().astype(np.int32)
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(t_min, t_max, steps):
        f1 = f1_score(y, (p > t).astype(np.int32), average='macro')
        if f1 > best_f1: best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


# -------------------------------------------------------------------
# Loss builders
# -------------------------------------------------------------------

def loss_x(batch, out, posw_x, cfg: TrainConfig):
    logits = out['x_logit']  # [E,1]
    y = batch[('ap', 'serves', 'user')].y
    # label smoothing
    eps = cfg.eps_smooth
    y_smooth = y * (1.0 - eps) + 0.5 * eps
    # per-sample BCE
    bce_vec = F.binary_cross_entropy_with_logits(
        logits.view(-1), y_smooth.view(-1), reduction='none', pos_weight=posw_x)
    # create w_hard for hard negative mining
    with torch.no_grad():
        ei = batch[('ap', 'serves', 'user')].edge_index[0]  # [E]
        logits = logits.view(-1)
        y = y.view(-1).long()
        w_hard = torch.ones_like(logits)
        for a in ei.unique():
            mask = (ei == a)
            neg = (~y[mask].bool()).nonzero(as_tuple=True)[0]
            if len(neg) > 0:
                topk = logits[mask][neg].topk(min(3, len(neg))).indices
                w_hard[mask.nonzero().view(-1)[neg[topk]]] = 2.0
    # utility weights
    w = edge_util_weights(batch, gain_col=cfg.gain_col_x)
    bce = (bce_vec * w * w_hard).mean()

    # RF cap penalty
    rf_pen = rf_penalty_from_logits(batch, logits, cfg.rf_cap)
    return bce, rf_pen


def loss_tau(batch, out, w0_w1):
    if 'tau_logit' not in out or 'ap' not in batch.node_types or 'y' not in batch['ap']:
        return torch.tensor(0.0, device=out['x_logit'].device)
    logits = out['tau_logit'].view(-1)
    y = batch['ap'].y.view(-1)
    vec = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    w0, w1 = w0_w1
    w = torch.where(y == 1, w1, w0)
    return (vec * w).mean()


def loss_s(batch, out, w0_w1):
    if 's_logit' not in out or 'target' not in batch.node_types or 'y' not in batch['target']:
        return torch.tensor(0.0, device=out['x_logit'].device)
    logits = out['s_logit'].view(-1)
    y = batch['target'].y.view(-1)
    vec = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    w0, w1 = w0_w1
    w = torch.where(y == 1, w1, w0)
    return (vec * w).mean()


# in case of focal loss: we may use for tau and/or s
def focal_bce_logits(logits, targets, alpha_pos=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = p * targets + (1 - p) * (1 - targets)
    alpha_t = targets * alpha_pos + (1 - targets) * (1 - alpha_pos)
    return (alpha_t * (1 - pt).pow(gamma) * ce).mean()


# --- to use:
# l_tau = focal_bce_logits(out['tau_logit'].view(-1), batch['ap'].y.view(-1),
#                          alpha_pos=0.1, gamma=2.0)  # negatives (tau=0) get 0.9
# l_s   = focal_bce_logits(out['s_logit'].view(-1), batch['target'].y.view(-1),
#                          alpha_pos=0.25, gamma=2.0)
# insteal of l_tau = loss_tau(batch, out, w0_w1)


def loss_edge_bce(batch, logits, rel, pos_weight=None, eps=0.0):
    y = batch[rel].y.view(-1)  # [E]
    y_s = y * (1 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(
        logits.view(-1), y_s, reduction='mean', pos_weight=pos_weight
    )


# -------------------------------------------------------------------
# Validation set loss
# -------------------------------------------------------------------
@torch.no_grad()
def eval_epoch(model, loader, cfg, posw_x, tau_w0w1, s_w0w1, posw_ytx, posw_yrx, device):
    model.eval()
    tot = x = tau = s = ytx = yrx = 0.0
    n = 0
    for b in loader:
        b = b.to(device)
        out = model(b)

        # x + rf
        Lx, Lrf = loss_x(b, out, posw_x, cfg)
        # tau / s
        Ltau = loss_tau(b, out, tau_w0w1)
        Ls = loss_s(b, out, s_w0w1)
        # y_tx / y_rx
        Lytx = loss_edge_bce(b, out['ytx_logit'], ('ap', 'senses_tx', 'target'), pos_weight=posw_ytx,
                             eps=cfg.eps_smooth)
        Lyrx = loss_edge_bce(b, out['yrx_logit'], ('ap', 'senses_rx', 'target'), pos_weight=posw_yrx,
                             eps=cfg.eps_smooth)

        # coupling
        if 'tau_logit' in out:
            Lc = tau_coupling_penalty(b, out['x_logit'], out['tau_logit'])
        else:
            Lc = torch.tensor(0.0, device=device)

        Ltot = (Lx + (cfg.w_Ltau * Ltau) + (cfg.w_Ls * Ls) + (cfg.w_Lytx * Lytx) + (cfg.w_Lyrx * Lyrx)
                + (cfg.w_Lrf * Lrf) + (cfg.w_Lcoup * Lc))

        x += float(Lx.item())
        tau += float(Ltau.item())
        s += float(Ls.item())
        ytx += float(Lytx.item())
        yrx += float(Lyrx.item())
        tot += float(Ltot.item())
        n += 1

    n = max(n, 1)
    return tot / n, x / n, tau / n, s / n, ytx / n, yrx / n


# -------------------------------------------------------------------
# EarlyStopper helper
# -------------------------------------------------------------------
class EarlyStopper:
    def __init__(self, mode="min", patience=10, min_delta=0.0):
        assert mode in ("min", "max")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def improved(self, current):
        if self.best is None:
            return True
        if self.mode == "min":
            return (self.best - current) > self.min_delta
        else:  # "max"
            return (current - self.best) > self.min_delta

    def update(self, current):
        if self.improved(current):
            self.best = current
            self.bad_epochs = 0
            return True  # improved
        else:
            self.bad_epochs += 1
            return False  # no improvement

    def should_stop(self):
        return self.bad_epochs >= self.patience


def f1_combined(f1x, f1tau, f1s, w=(0.5, 0.3, 0.2)):
    return w[0] * f1x + w[1] * f1tau + w[2] * f1s




# -------------------------------------------------------------------
# Pareto checkpointing
# -------------------------------------------------------------------

EPS = 1e-6
PARETO_KEYS = ("f1x","f1tau","f1s", "f1ytx", "f1yrx")
def dominates(a, b, keys=PARETO_KEYS, eps=EPS):
    """Return True if a dominates b (>= on all, > on at least one)."""
    ge = all(a[k] >= b[k] - eps for k in keys)
    gt = any(a[k] >  b[k] + eps for k in keys)
    return ge and gt

def update_pareto(pareto_list, candidate, max_keep=5):
    """Insert candidate if not dominated; remove points it dominates; cap length."""
    # remove any points dominated by candidate
    pareto_list = [p for p in pareto_list if not dominates(candidate, p)]
    # if candidate is dominated by any, skip add
    if any(dominates(p, candidate) for p in pareto_list):
        return pareto_list, False
    # else add it
    pareto_list.append(candidate)
    # optional: sort by a tie-break (e.g., sum of metrics) and cap
    pareto_list.sort(key=lambda d: -(d["f1x"]+d["f1tau"]+d["f1s"]))  # simple heuristic
    return pareto_list[:max_keep], True




# -------------------------------------------------------------------
# Log file saving
# -------------------------------------------------------------------
import sys
class TeeLogger(object):
    """Log to both console and a file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

