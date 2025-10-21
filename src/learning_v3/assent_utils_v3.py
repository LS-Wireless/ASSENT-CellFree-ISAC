
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class AssentData:
    G_comm: np.ndarray      # [A,U]
    S_comm: np.ndarray      # [A,U,U]
    G_sens: np.ndarray      # [A,A,T]
    alpha: float            # trade-off parameter
    lambda_cu: np.ndarray   # [U]
    lambda_tg: np.ndarray   # [T]
    solution: object        # Solution object (from optimization_uitils)
    N_RF: int = 4           # number of RF chains per AP
    K_tx: int = 6
    K_rx: int = 6
    device: str = "cpu"




def normalize_gcomm(G: np.ndarray, clip: float = 3.0):
    G = np.asarray(G, np.float32)
    G_db = 10.0 * np.log10(G + 1e-12)
    # per-AP z-score (robust for NNConv)
    mean = G_db.mean(axis=1, keepdims=True)
    std  = G_db.std(axis=1, keepdims=True) + 1e-6
    Gn = (G_db - mean) / std
    # optional: clamp to avoid extreme outliers
    return np.clip(Gn, -clip, clip).astype(np.float32)




def normalize_gsens(Gs: np.ndarray, eps: float = 1e-20, clip: float = 3.0):
    """
    Gs: [A, A, T] in linear scale (very small).
    """
    Gs = np.asarray(Gs, np.float32)
    Gs_db = 10.0 * np.log10(Gs + eps)
    mu = Gs_db.mean()
    sd = Gs_db.std() + 1e-6
    Gsn = (Gs_db - mu) / sd
    return np.clip(Gsn, -clip, clip).astype(np.float32)






def ap_sensing_features(Gsn, topk=5, clip=3.0, zscore_cols=True):
    """
    Z: [A, A, T] already in ~[-3,3]
    Returns ap_feat: [A, feat] where feat is 4 or 6 (see below).
    """
    A, A2, T = Gsn.shape
    assert A == A2
    tx_view = Gsn.reshape(A, A*T)                 # tx=a row over (rx,t)
    rx_view = np.transpose(Gsn, (1,0,2)).reshape(A, A*T)  # rx=a row over (tx,t)

    # hard max & log-sum-exp (soft coverage)
    tx_max = tx_view.max(axis=1)                # [A]
    rx_max = rx_view.max(axis=1)                # [A]

    def lse(v):
        m = v.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(v - m).sum(axis=1, keepdims=True) + 1e-12)).squeeze(1)
    tx_lse = lse(tx_view)                       # [A]
    rx_lse = lse(rx_view)                       # [A]

    # top-k mean (optional)
    K = min(topk, A*T)
    tx_topk = np.partition(tx_view, -K, axis=1)[:, -K:].mean(axis=1)
    rx_topk = np.partition(rx_view, -K, axis=1)[:, -K:].mean(axis=1)

    # choose 4 dims to stay lean (add the last two if you want 6 dims)
    cols = [tx_max, rx_max, tx_lse, rx_lse, tx_topk, rx_topk]     # optional: (+ tx_topk, rx_topk)
    feat = np.stack(cols, axis=1).astype(np.float32)

    if zscore_cols:
        # per-feature z-score across APs (per graph), then clip again to [-3,3]
        mu = feat.mean(axis=0, keepdims=True)
        sd = feat.std(axis=0, keepdims=True) + 1e-6
        feat = (feat - mu) / sd
        feat = np.clip(feat, -clip, clip)
    return feat  # shape [A, 4] (or [A,6] if include topk means)







def preprocess_scomm(S, standardize=False):
    """
    S: [A, U, U] (float32)
    Returns S_proc with per-AP normalization and/or standardization.
    """
    S = S.astype(np.float32)
    # symmetrize per-AP and take magnitude
    S = 0.5*(S + np.transpose(S, (0,2,1)))
    S = np.abs(S)
    A, U, _ = S.shape
    if standardize:
        Sf = S.reshape(A, -1)
        mean = Sf.mean(axis=1, keepdims=True)
        std  = Sf.std(axis=1, keepdims=True) + 1e-6
        Sn = ((Sf - mean)/std).reshape(A, U, U)
        Sn = np.clip(Sn, -6.0, 6.0).astype(np.float32)
    else:
        # normalize each AP's matrix to have max=1
        Smax = S.max(axis=(1, 2), keepdims=True) + 1e-6
        Sn = S / Smax
        Sn = np.clip(Sn, 0.0, 1.0).astype(np.float32)
    return Sn






def per_ap_rank01(Gn):
    # 1.0 = strongest per AP
    A, U = Gn.shape
    ranks = np.argsort(np.argsort(-Gn, axis=1), axis=1)  # 0 strongest
    return 1.0 - ranks/(U-1+1e-9)





def ap_user_edge_feats(G_comm, S_comm, topk=3):
    """
    Inputs:
      G_comm [A,U] linear -> we normalize inside
      S_comm [A,U,U]
    Returns:
      edge_attr [E, F] for E=A*U edges, features per (a,u) in order a-major
    """
    Gn = normalize_gcomm(G_comm)   # [A,U]
    S  = preprocess_scomm(S_comm, standardize=False)            # [A,U,U]
    A, U = Gn.shape

    # interference field per (a,u,v)
    Interf = S * Gn[:, None, :]                   # [A,U,U]
    # sum, max, topk-mean excluding self
    mask = np.ones((A, U, U), dtype=bool)
    np.fill_diagonal(mask.reshape(A, U*U), False)  # clear (u==v) per AP

    I_masked = np.where(mask, Interf, -np.inf)
    interf_sum = (np.where(mask, Interf, 0.0)).sum(axis=2)                         # [A,U]
    interf_max = np.max(I_masked, axis=2)                                     # [A,U]
    # top-k mean
    part = np.partition(I_masked, -topk, axis=2)[:, :, -topk:]              # [A,U,topk]
    interf_topk_mean = np.where(np.isfinite(part), part, 0.0).mean(axis=2)    # [A,U]
    # spread (std) on valid entries
    valid = np.where(mask, Interf, 0.0)
    spread = valid.std(axis=2)                                                # [A,U]
    rank01 = per_ap_rank01(Gn)                                                # [A,U]

    # stack features in this order:
    # [gain_norm, rank01, interf_sum, interf_max, interf_topk_mean, spread]
    feats = np.stack([Gn, rank01, interf_sum, interf_max, interf_topk_mean, spread], axis=2)  # [A,U,6]
    return feats.astype(np.float32)





def ap_target_edge_feats(Gsn, topk=3):
    """
    Build per-(ap,target) summaries from G_sens z-scored tensor Gsn [A_tx, A_rx, T].
    Returns:
      tx_feat [A,T,3]: for ap as Tx, aggregate over rx (max, lse, topk_mean)
      rx_feat [A,T,3]: for ap as Rx, aggregate over tx (max, lse, topk_mean)
    """
    A, A2, T = Gsn.shape
    assert A == A2
    # Tx view: for each (a_tx, t), pool over rx
    tx_view = np.transpose(Gsn, (0, 2, 1)).reshape(A, T, A)  # [A_tx, T, A_rx]
    # Rx view: for each (a_rx, t), pool over tx
    rx_view = np.transpose(Gsn, (1, 2, 0)).reshape(A, T, A)  # [A_rx, T, A_tx]

    def lse(x, axis=-1):
        m = x.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True) + 1e-12)).squeeze(axis)

    def topk_mean(x, k, axis=-1):
        k = min(k, x.shape[axis])
        part = np.partition(x, -k, axis=axis)[..., -k:]
        return part.mean(axis=axis)

    tx_max = tx_view.max(axis=2)
    rx_max = rx_view.max(axis=2)
    tx_lse = lse(tx_view, axis=2)
    rx_lse = lse(rx_view, axis=2)
    tx_kmn = topk_mean(tx_view, topk, axis=2)
    rx_kmn = topk_mean(rx_view, topk, axis=2)

    tx_feat = np.stack([tx_max, tx_lse, tx_kmn], axis=2).astype(np.float32)  # [A,T,3]
    rx_feat = np.stack([rx_max, rx_lse, rx_kmn], axis=2).astype(np.float32)  # [A,T,3]

    # Optional per-graph column-wise z-score for stability:
    def col_zscore(F):  # F [A,T,3]
        Ff = F.reshape(-1, F.shape[-1])
        mu = Ff.mean(axis=0, keepdims=True); sd = Ff.std(axis=0, keepdims=True) + 1e-6
        Fn = (Ff - mu) / sd
        return np.clip(Fn, -3.0, 3.0).reshape(F.shape).astype(np.float32)
    return col_zscore(tx_feat), col_zscore(rx_feat)





def data_loader(nsamps=10000):
    import os
    import json
    import src.utils.library as lib

    CONSOLE_RUN = False
    cwd = os.getcwd()
    console_run = cwd + '/src/optimization/exp1_baseline' if CONSOLE_RUN else '../optimization/exp1_baseline'

    exp1_run_id = 'run_01'
    exp1_folder_path = os.path.join(console_run, exp1_run_id)
    metadata_path = os.path.join(exp1_folder_path, 'metadata.json')
    with open(metadata_path) as f:
        exp1_metadata = json.load(f)
    num_parts = exp1_metadata['config']['num_parts_to_save']

    exp1_results = []
    for i in range(num_parts):
        filename = f'2025-10-03_results_p{i + 1}_of_{num_parts}.pkl'
        file_path = os.path.join(exp1_folder_path, filename)
        exp1_results += lib.load_results(file_path)

    console_run = cwd + '/src/optimization/exp2_data_gen' if CONSOLE_RUN else '../optimization/exp2_data_gen'
    exp2_run_id = 'run_01'
    exp2_folder_path = os.path.join(console_run, exp2_run_id)
    file_path = os.path.join(exp2_folder_path, '2025-10-13-13-25_results.pkl')
    exp2_results = lib.load_results(file_path)
    lib.print_log(tag='LOAD', message=f"Loaded exp2 results from '{exp2_folder_path}'")

    G_comm_list = [entry["G_comm"] for entry in exp1_results]
    S_comm_list = [entry["S_comm"] for entry in exp1_results]
    G_sens_list = [entry["G_sens"] for entry in exp1_results]

    alpha_list = [entry["alpha"] for entry in exp2_results]
    lambda_cu_list = [entry["lambda_cu"] for entry in exp2_results]
    lambda_tg_list = [entry["lambda_tg"] for entry in exp2_results]
    solution_list = [entry["solution"] for entry in exp2_results]


    result_lists = {'G_comm': G_comm_list[:nsamps], 'S_comm': S_comm_list[:nsamps], 'G_sens': G_sens_list[:nsamps],
                    'alpha': alpha_list[:nsamps], 'lambda_cu': lambda_cu_list[:nsamps], 'lambda_tg': lambda_tg_list[:nsamps],
                    'solution': solution_list[:nsamps], 'N_RF': exp1_metadata['NetworkParams']['N_RF']}
    return result_lists




def to_cpu_fp32(data):
    # Move everything to CPU float32 / int64 to save space & be portable
    for ntype in data.node_types:
        for k in list(data[ntype].keys()):
            t = data[ntype][k]
            if torch.is_tensor(t):
                if t.dtype.is_floating_point: data[ntype][k] = t.to('cpu', dtype=torch.float32)
                elif t.dtype == torch.int64:  data[ntype][k] = t.cpu()
                else:                         data[ntype][k] = t.cpu().to(torch.int64)
    for rel in data.edge_types:
        for k in list(data[rel].keys()):
            t = data[rel][k]
            if torch.is_tensor(t):
                if k == "edge_index":           data[rel][k] = t.to('cpu', dtype=torch.int64)
                elif t.dtype.is_floating_point: data[rel][k] = t.to('cpu', dtype=torch.float32)
                else:                           data[rel][k] = t.cpu()
    return data

