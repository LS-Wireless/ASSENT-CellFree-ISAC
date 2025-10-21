# assent_run.py

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import src.utils.library as lib
import time
import os


# -------------------------------------------------------------------
# Config
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
    w_rf: float = 0.2  # weight for RF penalty
    w_couple: float = 0.2  # weight for x–tau coupling penalty
    w_tau: float = 1.0  # weight for tau BCE
    w_s: float = 1.0  # weight for s BCE
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
    p  = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = p*targets + (1-p)*(1-targets)
    alpha_t = targets*alpha_pos + (1-targets)*(1-alpha_pos)
    return (alpha_t * (1-pt).pow(gamma) * ce).mean()
# --- to use:
# l_tau = focal_bce_logits(out['tau_logit'].view(-1), batch['ap'].y.view(-1),
#                          alpha_pos=0.1, gamma=2.0)  # negatives (tau=0) get 0.9
# l_s   = focal_bce_logits(out['s_logit'].view(-1), batch['target'].y.view(-1),
#                          alpha_pos=0.25, gamma=2.0)
# insteal of l_tau = loss_tau(batch, out, w0_w1)




# -------------------------------------------------------------------
# Validation set loss
# -------------------------------------------------------------------
@torch.no_grad()
def eval_epoch(model, loader, cfg, posw_x, tau_w0w1, s_w0w1, device):
    model.eval()
    tot = x = tau = s = rf = couple = 0.0
    n = 0
    for b in loader:
        b = b.to(device)
        out = model(b)

        # x + rf
        Lx, Lrf = loss_x(b, out, posw_x, cfg)
        # tau / s
        Ltau = loss_tau(b, out, tau_w0w1)
        Ls = loss_s(b, out, s_w0w1)

        # coupling
        if 'tau_logit' in out:
            Lc = tau_coupling_penalty(b, out['x_logit'], out['tau_logit'])
        else:
            Lc = torch.tensor(0.0, device=device)

        Ltot = Lx + cfg.w_rf*Lrf + cfg.w_couple*Lc + Ltau + Ls

        x += float(Lx.item()); rf += float(Lrf.item())
        tau += float(Ltau.item()); s += float(Ls.item()); couple += float(Lc.item())
        tot += float(Ltot.item()); n += 1

    n = max(n, 1)
    return tot / n, x / n, tau / n, s / n, rf / n, couple / n



# -------------------------------------------------------------------
# EarlyStopper helper
# -------------------------------------------------------------------
class EarlyStopper:
    def __init__(self, mode="min", patience=10, min_delta=0.0):
        assert mode in ("min","max")
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
            return True   # improved
        else:
            self.bad_epochs += 1
            return False  # no improvement

    def should_stop(self):
        return self.bad_epochs >= self.patience

def f1_combined(f1x, f1tau, f1s, w=(0.5, 0.3, 0.2)):
    return w[0]*f1x + w[1]*f1tau + w[2]*f1s


# -------------------------------------------------------------------
# Training / evaluation
# -------------------------------------------------------------------
def run_training(model, train_loader, val_loader, cfg: TrainConfig,
                 save_model_state: bool = False, save_path: str = "checkpoints", save_every=1):
    device = cfg.device
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
    warmup_epochs = cfg.warmup_epochs

    lib.print_log(tag='RUN', message=f'Training {model.__class__.__name__} on {cfg.device} for {cfg.epochs} epochs')

    # precompute weights
    posw_x = compute_pos_weight_x(train_loader, device)
    tau_w0w1 = class_weights_node(train_loader, 'ap', device)
    s_w0w1 = class_weights_node(train_loader, 'target', device)

    from torch import amp
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    scaler = amp.GradScaler(device_type, enabled=(cfg.amp and device_type == 'cuda'))

    # --- Track best metrics for all outputs ---
    best = {
        'f1x': -1.0, 'f1tau': -1.0, 'f1s': -1.0,
        'state': None,
        'thr_x': 0.5, 'thr_tau': 0.5, 'thr_s': 0.5
    }

    stopper = EarlyStopper(mode=cfg.es_mode, patience=cfg.es_patience, min_delta=cfg.es_min_delta)

    # before training loop
    history = {
        "epoch": [], "lr": [],
        # train losses
        "train_loss_total": [], "train_loss_x": [], "train_loss_tau": [],
        "train_loss_s": [], "train_loss_rf": [], "train_loss_couple": [],
        # val loss (same objective as train for apples-to-apples)
        "val_loss_total": [], "val_loss_x": [], "val_loss_tau": [],
        "val_loss_s": [], "val_loss_rf": [], "val_loss_couple": [],
        # val metrics (F1s and thresholds)
        "val_f1x_u": [], "val_f1tau_m": [], "val_f1s_m": [],
        "thr_x": [], "thr_tau": [], "thr_s": []
    }
    lib.print_log(tag='RUN', message='Starting training epochs...')
    for ep in range(1, cfg.epochs + 1):
        train_start = time.time()
        # ----------------- train -----------------
        model.train()
        tr_losses = {'x': 0.0, 'rf': 0.0, 'tau': 0.0, 's': 0.0, 'coup': 0.0}
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device_type, enabled=cfg.amp):
                out = model(batch)

                # x + rf
                bce_x, rf_pen = loss_x(batch, out, posw_x, cfg)
                # tau / s
                l_tau = loss_tau(batch, out, tau_w0w1)
                l_s = loss_s(batch, out, s_w0w1)

                # coupling (requires tau logits)
                if 'tau_logit' in out:
                    l_coup = tau_coupling_penalty(batch, out['x_logit'], out['tau_logit'])
                else:
                    l_coup = torch.tensor(0.0, device=device)
                # loss = bce_x + cfg.w_rf * rf_pen + cfg.w_couple * l_coup + cfg.w_tau * l_tau + cfg.w_s * l_s
                # light clamp each step before loss usage
                model.log_var_x.data.clamp_(-5.0, 3.0)
                model.log_var_tau.data.clamp_(-5.0, 3.0)
                model.log_var_s.data.clamp_(-5.0, 3.0)
                # Uncertainty-weighted sum:
                loss_multi = (
                        torch.exp(-model.log_var_x) * bce_x + model.log_var_x +
                        torch.exp(-model.log_var_tau) * (cfg.w_tau * l_tau) + model.log_var_tau +
                        torch.exp(-model.log_var_s) * (cfg.w_s * l_s) + model.log_var_s
                )
                # Then add your structural penalties with fixed weights:
                loss = loss_multi + cfg.w_rf * rf_pen + cfg.w_couple * l_coup

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            tr_losses['x'] += bce_x.item()
            tr_losses['rf'] += rf_pen.item()
            tr_losses['tau'] += float(l_tau.item()) if torch.is_tensor(l_tau) else 0.0
            tr_losses['s'] += float(l_s.item()) if torch.is_tensor(l_s) else 0.0
            tr_losses['coup'] += l_coup.item()
        current_lr = opt.param_groups[0]['lr']
        train_time = time.time() - train_start
        ntr = max(1, len(train_loader))
        print(f"\nEpoch {ep:03d} | train-loss={loss.detach().item() / ntr:.4f} | "
              f"x-loss={tr_losses['x'] / ntr:.4f} | "
              f"tau-loss={tr_losses['tau'] / ntr:.4f} | s-loss={tr_losses['s'] / ntr:.4f} | "
              f"lr={current_lr:.2e} | train-time={train_time:.2f}s")

        # ----------------- validation (threshold search) -----------------
        eval_start = time.time()
        model.eval()
        thr_x, best_f1x, (Pu, Ru) = best_threshold_utility_f1_x(val_loader, model, device, cfg.gain_col_x,
                                                                cfg.thresh_min, cfg.thresh_max, cfg.thresh_steps)
        thr_tau, best_f1tau = best_threshold_macro_f1_nodes(val_loader, model, device, ntype='ap',
                                                            t_min=0.1, t_max=0.9, steps=33)
        thr_s, best_f1s = best_threshold_macro_f1_nodes(val_loader, model, device, ntype='target',
                                                        t_min=0.1, t_max=0.9, steps=33)

        val_tot, val_x, val_tau, val_s, val_rf, val_couple = eval_epoch(model, val_loader, cfg, posw_x,
                                                                        tau_w0w1, s_w0w1, cfg.device)

        eval_time = time.time() - eval_start
        print(f" Validate | val-loss={val_tot:.4f} | x-F1={best_f1x:.3f} @ {thr_x:.2f} (P={Pu:.3f}, R={Ru:.3f}) "
              f"| tau-F1={best_f1tau:.3f} @ {thr_tau:.2f} | s-F1={best_f1s:.3f} @ {thr_s:.2f} | eval-time={eval_time:.2f}s")

        # keep best by x Utility-F1 (you can change to a weighted combo)
        if best_f1x > best['f1x']:
            best.update({'f1x': best_f1x, 'f1tau': best_f1tau, 'f1s': best_f1s,
                         'state': model.state_dict(),
                         'thr_x': thr_x, 'thr_tau': thr_tau, 'thr_s': thr_s})
            if save_model_state and (ep % save_every == 0):
                os.makedirs(save_path, exist_ok=True)
                ckpt_path = os.path.join(save_path, f"best_ep{ep:03d}.pt")
                torch.save(model.state_dict(), ckpt_path)
                lib.print_log(tag='SAVE', message=f'Saved checkpoint to {ckpt_path} at epoch {ep}')

        # NEW: pick early-stop metric value
        if cfg.es_metric == "val_loss_total":
            es_value = val_tot
            es_mode = "min"
        elif cfg.es_metric == "val_loss_total_raw":
            es_value = val_tot  # same here if you're already computing raw; otherwise compute raw variant
            es_mode = "min"
        elif cfg.es_metric == "f1_combo":
            es_value = f1_combined(best_f1x, best_f1tau, best_f1s)
            es_mode = "max"
        else:
            # fallback to val loss
            es_value = val_tot
            es_mode = "min"

        # ensure stopper mode aligns
        stopper.mode = es_mode

        # update stopper
        improved = stopper.update(es_value)
        if improved:
            print(f"  ✓ EarlyStop monitor improved: {cfg.es_metric}={es_value:.4f}")
            lib.print_log(tag='TRAIN', message=f"EarlyStop monitor improved: {cfg.es_metric}={es_value:.4f}")
        else:
            lib.print_log(tag='TRAIN', message=f"No improvement on {cfg.es_metric} for {stopper.bad_epochs}/{cfg.es_patience} epochs")

        # check stop condition
        if stopper.should_stop():
            lib.print_log(tag='TRAIN', message=f"Early stopping triggered on {cfg.es_metric}.")
            break
        # --- History ---
        history["epoch"].append(ep)
        history["lr"].append(current_lr)
        history["train_loss_total"].append((loss / ntr).detach().item())
        history["train_loss_x"].append(float(tr_losses['x'] / ntr))
        history["train_loss_tau"].append(float(tr_losses['tau'] / ntr))
        history["train_loss_s"].append(float(tr_losses['s'] / ntr))
        history["train_loss_rf"].append(float(tr_losses['rf'] / ntr))
        history["train_loss_couple"].append(float(tr_losses['coup'] / ntr))
        history["val_loss_total"].append(float(val_tot))
        history["val_loss_x"].append(float(val_x))
        history["val_loss_tau"].append(float(val_tau))
        history["val_loss_s"].append(float(val_s))
        history["val_loss_rf"].append(float(val_rf))
        history["val_loss_couple"].append(float(val_couple))
        history["val_f1x_u"].append(float(best_f1x))
        history["val_f1tau_m"].append(float(best_f1tau))
        history["val_f1s_m"].append(float(best_f1s))
        history["thr_x"].append(float(thr_x))
        history["thr_tau"].append(float(thr_tau))
        history["thr_s"].append(float(thr_s))

        # step LR scheduler
        if ep <= warmup_epochs:
            for g in opt.param_groups:
                g['lr'] = cfg.lr_min + (cfg.lr - cfg.lr_min) * ep / warmup_epochs
        else:
            sched.step()

    lib.print_log(tag='RUN', message='Finished training!\n')
    # load best & return summary
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    summary = {'best_f1x': best['f1x'], 'best_f1tau': best['f1tau'], 'best_f1s': best['f1s'],
               'thr_x': best['thr_x'], 'thr_tau': best['thr_tau'], 'thr_s': best['thr_s']
               }
    return model, summary, history


# -------------------------------------------------------------------
# Main function to run the experiment
# -------------------------------------------------------------------
if __name__ == "__main__":

    from assent_data_v3 import GraphDataset
    from assent_utils_v3 import data_loader
    from torch_geometric.loader import DataLoader
    from assent_model_v3 import ASSENTGNN

    nsamps = 10000
    SEED = 42
    train_epochs = 150
    nhidden = 128
    nlayers = 3
    lr = 1e-3
    drop = 0.2
    batch_size = 16
    SAVE_MODEL = True

    results = data_loader(nsamps=nsamps)
    train_cfg = TrainConfig(epochs=train_epochs, lr=lr, batch_size=batch_size, device="cpu", rf_cap=results["N_RF"])

    # split
    N = nsamps
    idx = np.random.RandomState(0).permutation(N)
    ntr = int(0.8 * N)
    nva = int(0.1 * N)
    tr_idx, va_idx = idx[:ntr], idx[ntr:ntr + nva]
    te_idx = idx[ntr + nva:]

    ds = GraphDataset(results["G_comm"], results["S_comm"], results["G_sens"],
                      results["alpha"], results["lambda_cu"], results["lambda_tg"],
                      results["solution"], N_RF=results["N_RF"], device="cpu")

    from torch.utils.data import Subset

    tr_set, va_set, te_set = Subset(ds, tr_idx.tolist()), Subset(ds, va_idx.tolist()), Subset(ds, te_idx.tolist())

    lib.print_log(tag='RUN',
                  message=f"Training set size: {len(tr_set)}; Validation set size: {len(va_set)}; Test set size: {len(te_set)}")
    tr_loader = DataLoader(tr_set, batch_size=train_cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=train_cfg.batch_size, shuffle=False)
    te_loader = DataLoader(te_set, batch_size=train_cfg.batch_size, shuffle=False)

    model = ASSENTGNN(hidden=nhidden, layers=nlayers, drop=drop).to(train_cfg.device)

    model, summary, history = run_training(model, tr_loader, va_loader, train_cfg,
                                           save_model_state=True, save_path="checkpoints", save_every=10)


    if SAVE_MODEL:
        from datetime import datetime
        import json

        # === Create timestamped save directory ===
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        save_dir = f"checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        # === Save model weights ===
        model_path = os.path.join(save_dir, timestamp+"_best_model_fully_trained.pt")
        torch.save(model.state_dict(), model_path)
        lib.print_log(tag='SAVE', message=f"Saved model weights to {model_path}")

        # --- Save Summary as JSON ---
        hist_json = os.path.join(save_dir, timestamp+"_history.json")
        with open(hist_json, "w") as f:
            json.dump(history, f, indent=2)
        lib.print_log(tag='SAVE', message=f"Saved history JSON to {hist_json}")

        # === Save summary ===
        summary["model_path"] = model_path  # add model path reference
        summary["history_path"] = hist_json
        summary_path = os.path.join(save_dir, timestamp+"_summary.json")

        # ensure everything in summary is serializable
        safe_summary = {}
        for k, v in summary.items():
            if isinstance(v, torch.Tensor):
                safe_summary[k] = v.detach().cpu().item()
            elif isinstance(v, (list, tuple)):
                safe_summary[k] = [float(x.detach().cpu().item()) if isinstance(x, torch.Tensor) else x for x in v]
            else:
                safe_summary[k] = v

        with open(summary_path, "w") as f:
            json.dump(safe_summary, f, indent=2)
        lib.print_log(tag='SAVE', message=f"Saved summary to {summary_path}")



