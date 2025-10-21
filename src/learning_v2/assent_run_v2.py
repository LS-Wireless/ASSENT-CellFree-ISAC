# file: x_only_train.py
import numpy as np, torch, os
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from assent_data_v2 import XOnlyDataset, data_loader
from assent_model_v2 import XOnlyGNN
import src.utils.library as lib
import time
from datetime import datetime


def compute_pos_weight(loader, device):
    pos = neg = 0
    for b in loader:
        y = b[('ap', 'serves', 'user')].y.view(-1)
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    w = float(neg / max(pos, 1))
    return torch.tensor(w, device=device)




@torch.no_grad()
def f1_from_logits(logits, labels, thr=0.5):
    p = torch.sigmoid(logits).view(-1);
    y = labels.view(-1)
    yhat = (p > thr).float()
    tp = (yhat.eq(1) & y.eq(1)).sum().item()
    fp = (yhat.eq(1) & y.eq(0)).sum().item()
    fn = (yhat.eq(0) & y.eq(1)).sum().item()
    prec = tp / (tp + fp + 1e-9);
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return f1





@torch.no_grad()
def best_thresh_for_f1(val_loader, model, device):
    """
    Computes the best threshold for F1 on the validation set.
    """
    model.eval()
    ps, ys = [], []
    for b in val_loader:
        b = b.to(device)
        logits = model(b)['x_logit']
        p = torch.sigmoid(logits).view(-1).cpu()
        y = b[('ap', 'serves', 'user')].y.view(-1).cpu()
        ps.append(p)
        ys.append(y)
    if not ps:
        return 0.5, 0.0  # safety
    p = torch.cat(ps).numpy()
    y = torch.cat(ys).numpy()

    import numpy as np
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.2, 0.8, 31):
        yhat = (p > t).astype(np.int32)
        tp = ((yhat == 1) & (y == 1)).sum()
        fp = ((yhat == 1) & (y == 0)).sum()
        fn = ((yhat == 0) & (y == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1




def edge_util_weights(b):
    # build per-edge weights w_{a,u}
    ei = b[('ap', 'serves', 'user')].edge_index
    a_idx, u_idx = ei[0], ei[1]
    alpha = b['ap'].x[0, 0]  # same alpha per graph
    lam_u = b['user'].x[:, 1]  # [U]
    Gnorm = b[('ap', 'serves', 'user')].edge_attr[:, 0]  # your normalized gain in col 0
    w = (1.0 - alpha) * lam_u[u_idx] * Gnorm  # [E]
    w = (w / (w.mean() + 1e-8)).clamp_(0.25, 4.0)  # tame extremes
    return w





def run_training(G_comm_list, S_comm_list, G_sens_list, alpha_list, lambda_cu_list, solution_list,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 epochs=20, batch_size=8, hidden=64, layers=2, lr=3e-3, drop=0.1):
    lib.print_log(tag='RUN', message=f"Running training with {len(G_comm_list)} graphs")
    # split
    N = len(G_comm_list)
    idx = np.random.RandomState(0).permutation(N)
    ntr = int(0.8 * N)
    nva = int(0.1 * N)
    tr_idx, va_idx = idx[:ntr], idx[ntr:ntr + nva]
    te_idx = idx[ntr + nva:]

    ds = XOnlyDataset(G_comm_list, S_comm_list, G_sens_list, alpha_list, lambda_cu_list, solution_list, device=device)
    from torch.utils.data import Subset
    tr_set, va_set, te_set = Subset(ds, tr_idx.tolist()), Subset(ds, va_idx.tolist()), Subset(ds, te_idx.tolist())

    lib.print_log(tag='RUN', message=f"Training set size: {len(tr_set)}; Validation set size: {len(va_set)}; Test set size: {len(te_set)}")
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_set, batch_size=batch_size, shuffle=False)

    model = XOnlyGNN(hidden=hidden, layers=layers, drop=drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    posw = compute_pos_weight(tr_loader, device)

    best = None
    lib.print_log(tag='RUN', message=f"Starting training for {epochs} epochs...\n")
    for ep in range(1, epochs + 1):
        train_start = time.time()
        model.train()
        tr_loss = 0.0
        for b in tr_loader:
            opt.zero_grad()
            out = model(b)
            logits_x = out['x_logit']  # [E,1]
            logits_tau = out['tau_logit']  # [A,1]

            y_x_orig = b[('ap','serves','user')].y  # [E,1]
            SMOOTHING = True
            eps = 0.05
            y_x_smooth = y_x_orig * (1-eps) + (eps / 2)
            y_x = y_x_smooth if SMOOTHING else y_x_orig

            # bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=posw)
            w = edge_util_weights(b)
            bce_vec = F.binary_cross_entropy_with_logits(logits_x.view(-1), y_x.view(-1), reduction='none', pos_weight=posw)
            bce = (bce_vec * w).mean()

            # 2) RF-chain penalty
            probs = torch.sigmoid(logits_x).view(-1)  # [E]
            ei = b[('ap', 'serves', 'user')].edge_index  # [2, E]
            ap_of_edge = ei[0]  # [E]
            A = b['ap'].x.size(0)  # number of APs in this graph
            N_RF = 4  # <-- set from config or batch metadata: scalar (same cap for all APs) or a tensor [A] per-AP
            # sum predicted x per AP
            per_ap_sum = torch.zeros(A, device=probs.device).scatter_add_(0, ap_of_edge, probs)
            if np.isscalar(N_RF):
                rf_excess = (per_ap_sum - float(N_RF)).clamp(min=0)
                rf_pen = rf_excess.mean()
            elif isinstance(N_RF, np.ndarray) or isinstance(N_RF, list):
                rf_excess = (per_ap_sum - N_RF.to(probs.device)).clamp(min=0)
                rf_pen = rf_excess.mean()
            else:
                rf_pen = 0.0

            # --- tau coupling penalty (no tau labels needed) ---
            a_idx = b[('ap', 'serves', 'user')].edge_index[0]  # [E]
            xprob = torch.sigmoid(logits_x.view(-1))
            tauprob = torch.sigmoid(logits_tau.view(-1))
            # encourage x <= tau for each edge’s AP
            coup = (xprob - tauprob[a_idx]).clamp(min=0).mean()

            # --- total loss ---
            loss = bce + 0.2 * rf_pen + 0.2 * coup
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(len(tr_loader), 1)

        # eval
        model.eval()
        f1s = []
        vloss = 0.0
        with torch.no_grad():
            for b in va_loader:
                logits = model(b)['x_logit']
                y = b[('ap', 'serves', 'user')].y
                vloss += F.binary_cross_entropy_with_logits(logits, y, pos_weight=posw).item()
                f1s.append(f1_from_logits(logits, y))
        # vloss /= max(len(va_loader), 1)
        # f1 = np.mean(f1s) if f1s else 0.0
        # print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_loss={vloss:.4f} | x_F1={f1:.3f}")

        vloss /= max(len(va_loader), 1)
        mean_f1 = np.mean(f1s) if f1s else 0.0

        # find best threshold and F1 on the validation set
        best_t, best_f1 = best_thresh_for_f1(va_loader, model, device)

        train_time = time.time() - train_start
        print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_loss={vloss:.4f} "
              f"| best_F1={best_f1:.3f} @ thr={best_t:.2f} | train_time={train_time:.2f}s")

        best = {'state': model.state_dict()}

    # test
    if best: model.load_state_dict(best['state'])
    model.eval()
    f1s = []
    with torch.no_grad():
        for b in te_loader:
            f1s.append(f1_from_logits(model(b)['x_logit'], b[('ap', 'serves', 'user')].y))
    lib.print_log(tag='TEST', message=f"x_F1 = {np.mean(f1s):.3f}")
    lib.print_log(tag='RUN', message=f'Finished training!\n')
    return model







# ----------------------------------# Main script #---------------------------------- #
SEED = 42
train_epochs = 15
nhidden = 128
nlayers = 3
lr = 1e-3
drop = 0.2
SAVE_MODEL = False

np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

result_lists = data_loader(nsamps=2000)

model = run_training(
    G_comm_list=result_lists['G_comm'],
    S_comm_list=result_lists['S_comm'],
    G_sens_list=result_lists['G_sens'],
    alpha_list=result_lists['alpha'],
    lambda_cu_list=result_lists['lambda_cu'],
    solution_list=result_lists['solution'],
    device=device,
    epochs=train_epochs,
    batch_size=16,
    hidden=nhidden,
    layers=nlayers,
    lr=lr,
    drop=drop
)

# 5) Save the trained model (optional)
if SAVE_MODEL:
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    ckpt_path = os.path.join("checkpoints", timestamp+"_assent_gnn.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model checkpoint to: {ckpt_path}")

test_ds = XOnlyDataset(result_lists['G_comm'], result_lists['S_comm'], result_lists['G_sens'],
                       result_lists['alpha'], result_lists['lambda_cu'],
                       result_lists['solution'], device=device)
test_loader = DataLoader([test_ds[0]], batch_size=1, shuffle=False)  # single graph
model.eval()
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch)  # [E,1]
        probs = torch.sigmoid(logits['x_logit']).view(-1)  # [E]
        print("First 10 x-edge probabilities:", probs[:10].cpu().numpy())
        tau = torch.sigmoid(logits['tau_logit']).cpu().numpy().squeeze()
        print("Mean tau per AP:", tau.mean(), "min:", tau.min(), "max:", tau.max())
        break
