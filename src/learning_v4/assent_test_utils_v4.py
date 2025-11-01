

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss, confusion_matrix


def load_state_into_model(model, ckpt_path, device="cuda", strict=True, key="state_dict"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get(key, ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if (missing or unexpected) and strict:
        print("Warning: missing keys:", missing, "unexpected:", unexpected)
    model.to(device)
    return model




# Map task key -> hetero relation (or node type) that carries logits/labels
REL_MAP = {
    "x":   ("ap","serves","user"),
    "ytx": ("ap","senses_tx","target"),
    "yrx": ("ap","senses_rx","target"),
    "tau": "ap",
    "s":   "target",
}

def _store_has_attr(store, name: str) -> bool:
    # hasattr is safe with PyG storages; returns False if attr missing
    return hasattr(store, name)

@torch.no_grad()
def run_inference(model, loader, device, thr_dict=None, return_metrics=True):
    model.eval()
    prob_buf = {k: [] for k in REL_MAP}
    y_buf    = {k: [] for k in REL_MAP}  # may stay empty if labels absent

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)

        for key, rel in REL_MAP.items():
            logit_key = f"{key}_logit"
            if logit_key not in out:
                continue

            probs = torch.sigmoid(out[logit_key]).flatten().cpu()
            prob_buf[key].append(probs)

            # Try to fetch labels only if present
            store = batch[rel] if isinstance(rel, tuple) else batch[rel]
            if _store_has_attr(store, "y"):
                y_true = getattr(store, "y").flatten().cpu()
                y_buf[key].append(y_true)

    # Concatenate
    probs_cat = {k: (torch.cat(v) if len(v) else torch.empty(0)) for k, v in prob_buf.items()}
    y_cat     = {k: (torch.cat(v) if len(v) else None) for k, v in y_buf.items()}

    # Threshold
    preds_bin = {}
    for k, p in probs_cat.items():
        if p.numel() == 0:
            preds_bin[k] = torch.empty(0, dtype=torch.int32)
            continue
        thr = (thr_dict or {}).get(f"thr_{k}", 0.5)
        preds_bin[k] = (p > thr).to(torch.int32)

    # Optional metrics (only where labels exist)
    metrics = {}
    if return_metrics:
        for k in REL_MAP:
            y = y_cat[k]
            if y is None or y.numel() == 0:
                metrics[k] = {"f1": None, "prec": None, "rec": None}
                continue
            y_pred = preds_bin[k].numpy()
            y_true = y.numpy()
            metrics[k] = {
                "f1":  f1_score(y_true, y_pred, zero_division=0),
                "prec": precision_score(y_true, y_pred, zero_division=0),
                "rec": recall_score(y_true, y_pred, zero_division=0),
            }

    return probs_cat, preds_bin, y_cat, metrics





TASKS = ["x", "tau", "s", "ytx", "yrx"]

def summarize_inference(probs_cat, preds_bin, y_cat, keys=TASKS):
    """
    probs_cat, preds_bin, y_cat are the dicts returned by run_inference().
    Returns a list of rows (dicts) with metrics per variable.
    """
    rows = []
    for k in keys:
        y_true_t = y_cat.get(k, None)
        if y_true_t is None or (hasattr(y_true_t, "numel") and y_true_t.numel() == 0):
            rows.append({
                "var": k.upper(), "F1": None, "Precision": None, "Recall": None,
                "Brier": None, "TP": None, "FP": None, "FN": None, "TN": None, "N": 0
            })
            continue

        # to numpy
        y_true = y_true_t.numpy().astype(int)
        y_hat  = preds_bin[k].numpy().astype(int)
        p_hat  = probs_cat[k].numpy().astype(float)

        # metrics
        f1  = f1_score(y_true, y_hat, zero_division=0)
        pre = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        # Brier expects probabilities; protect against degenerate arrays
        brier = brier_score_loss(y_true, np.clip(p_hat, 1e-6, 1-1e-6)) if y_true.size else None

        # confusion matrix (handles all-zero/one safely)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()

        rows.append({
            "var": k.upper(),
            "F1": f1, "Precision": pre, "Recall": rec,
            "Brier": brier, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "N": int(y_true.size),
        })
    return rows

def print_metrics_table(rows):
    # Compact, readable console print
    hdr = f"{'Var':<5} {'N':>6} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Brier':>8}   {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    print(hdr)
    print("-"*len(hdr))
    for r in rows:
        def fmt(x, w, p=3):
            if x is None: return f"{'-':>{w}}"
            if isinstance(x, (int, np.integer)): return f"{x:>{w}d}"
            return f"{x:>{w}.{p}f}"
        line = (
            f"{r['var']:<5} {fmt(r['N'],6)} {fmt(r['F1'],7)} {fmt(r['Precision'],7)} {fmt(r['Recall'],7)} "
            f"{fmt(r['Brier'],8)}   {fmt(r['TP'],5)} {fmt(r['FP'],5)} {fmt(r['FN'],5)} {fmt(r['TN'],5)}"
        )
        print(line)






import src.utils.optimization_utils as opt


# ---------- Small helpers for splitting edges per-graph ----------
def _edge_mask_for_sample(ei, src0, src1, dst0, dst1):
    src, dst = ei[0], ei[1]
    return (src >= src0) & (src < src1) & (dst >= dst0) & (dst < dst1)

def _localize_edges(ei, mask, src_off, dst_off):
    src = (ei[0][mask] - src_off).cpu().numpy()
    dst = (ei[1][mask] - dst_off).cpu().numpy()
    return src, dst, mask.nonzero(as_tuple=False).view(-1)

# ---------- Main comparison function (Fix B: shuffle=False, cursor over test_idx) ----------
@torch.no_grad()
def milp_vs_assent(model, test_loader, test_idx, device, results, thr_dict=None):
    model.eval()
    recs, cursor = [], 0


    # thresholds (can pass summary dict with thr_x, thr_tau, thr_s, thr_ytx, thr_yrx)
    thr = {'thr_x':0.5, 'thr_tau':0.5, 'thr_s':0.5, 'thr_ytx':0.5, 'thr_yrx':0.5}
    if thr_dict: thr.update(thr_dict)

    for batch in test_loader:
        # number of graphs in this batch (use ptr length - 1)
        nS = int(batch['ap'].ptr.numel() - 1)

        batch = batch.to(device)
        out = model(batch)

        # Node probs
        tau_p = torch.sigmoid(out['tau_logit']).flatten()  # [sum_A]
        s_p   = torch.sigmoid(out['s_logit']).flatten()    # [sum_T]
        # Edge probs
        p_x   = torch.sigmoid(out['x_logit']).flatten()
        p_ytx = torch.sigmoid(out['ytx_logit']).flatten()
        p_yrx = torch.sigmoid(out['yrx_logit']).flatten()

        # ptrs (CPU) and edge_index (CPU)
        ap_ptr  = batch['ap'].ptr.cpu().numpy()
        usr_ptr = batch['user'].ptr.cpu().numpy()
        tgt_ptr = batch['target'].ptr.cpu().numpy()

        ei_s  = batch[('ap','serves','user')].edge_index.cpu()
        ei_tx = batch[('ap','senses_tx','target')].edge_index.cpu()
        ei_rx = batch[('ap','senses_rx','target')].edge_index.cpu()

        for s in range(nS):
            # Map to original dataset index using cursor over test_idx
            gid = int(test_idx[cursor + s])

            # slice bounds & sizes for this graph
            a0, a1 = ap_ptr[s],  ap_ptr[s+1]
            u0, u1 = usr_ptr[s], usr_ptr[s+1]
            t0, t1 = tgt_ptr[s], tgt_ptr[s+1]
            A = a1 - a0; U = u1 - u0; T = t1 - t0

            # ---- reconstruct predictions (threshold per-graph slice) ----
            tau_hat = (tau_p[a0:a1] > thr['thr_tau']).to(torch.int8).cpu().numpy()
            s_hat   = (s_p[t0:t1]   > thr['thr_s']).to(torch.int8).cpu().numpy()

            # X (ap-user)
            m_x = _edge_mask_for_sample(ei_s, a0, a1, u0, u1)
            i_x, j_x, idx_x = _localize_edges(ei_s, m_x, a0, u0)
            x_hat = np.zeros((A, U), dtype=np.int8)
            if idx_x.numel() > 0:
                x_hat[i_x, j_x] = (p_x[idx_x] > thr['thr_x']).to(torch.int8).cpu().numpy()

            # Ytx (ap-target)
            m_tx = _edge_mask_for_sample(ei_tx, a0, a1, t0, t1)
            i_tx, j_tx, idx_tx = _localize_edges(ei_tx, m_tx, a0, t0)
            ytx_hat = np.zeros((A, T), dtype=np.int8)
            if idx_tx.numel() > 0:
                ytx_hat[i_tx, j_tx] = (p_ytx[idx_tx] > thr['thr_ytx']).to(torch.int8).cpu().numpy()

            # Yrx (ap-target)
            m_rx = _edge_mask_for_sample(ei_rx, a0, a1, t0, t1)
            i_rx, j_rx, idx_rx = _localize_edges(ei_rx, m_rx, a0, t0)
            yrx_hat = np.zeros((A, T), dtype=np.int8)
            if idx_rx.numel() > 0:
                yrx_hat[i_rx, j_rx] = (p_yrx[idx_rx] > thr['thr_yrx']).to(torch.int8).cpu().numpy()

            # (Optional quick feasibility: cap per-target top-K, tau/yrx coupling) — can add here.

            # ---- compute utilities: MILP vs ASSENT ----
            params = {
                'G_comm': results['G_comm'][gid], 'S_comm': results['S_comm'][gid], 'G_sens': results['G_sens'][gid],
                'lambda_cu': results['lambda_cu'][gid], 'lambda_tg': results['lambda_tg'][gid],
                'alpha': results['alpha'][gid], 'interf_penalty': 0.01
            }
            sol_gt = results['solution'][gid]  # dict with 'x','tau','s','y_tx','y_rx'
            sol_pd = {'x': x_hat, 'tau': tau_hat, 's': s_hat, 'y_tx': ytx_hat, 'y_rx': yrx_hat}

            gt_out = opt.compute_milp_objective(params, sol_gt)
            U_gt = gt_out['obj_val']
            pd_out = opt.compute_milp_objective(params, sol_pd)
            U_pd = pd_out['obj_val']
            gap_pct = float(100.0 * (U_gt - U_pd) / (abs(U_gt) + 1e-9))

            recs.append({'sid': gid, 'U_gt': float(U_gt), 'U_pd': float(U_pd),
                         'gap_pct': gap_pct, 'alpha': float(params['alpha'])})

        cursor += nS  # advance by the number of graphs in this batch

    return recs

