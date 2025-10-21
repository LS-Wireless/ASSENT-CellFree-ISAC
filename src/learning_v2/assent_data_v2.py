# file: x_only_data.py
from dataclasses import dataclass
import numpy as np
import torch
from torch_geometric.data import HeteroData

@dataclass
class XOnlyInstance:
    G_comm: np.ndarray      # [A,U]
    S_comm: np.ndarray      # [A,U,U]
    G_sens: np.ndarray      # [A,A,T]
    alpha: float
    lambda_cu: np.ndarray   # [U]
    x: np.ndarray           # [A,U] labels (0/1)





# def normalize_gcomm(G):
#     G = np.asarray(G, dtype=np.float32)
#     G = np.log1p(G)
#     mx = G.max() if G.size else 1.0
#     return (G / (mx + 1e-6)).astype(np.float32)

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
    Returns Gs01 in ~[0,1] after dB + zscore + clipping.
    """
    Gs = np.asarray(Gs, np.float32)
    Gs_db = 10.0 * np.log10(Gs + eps)                 # -> dB
    mu = Gs_db.mean()
    sd = Gs_db.std() + 1e-6
    Gsn = (Gs_db - mu) / sd                              # z-score
    return np.clip(Gsn, -clip, clip).astype(np.float32)





def ap_sensing_features(Gsn, topk=5, clip=3.0, zscore_cols=True):
    """
    Z: [A, A, T] already in ~[-3,3]
    Returns ap_feat: [A, F] where F is 4 or 6 (see below).
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
    cols = [tx_max, rx_max, tx_lse, rx_lse, tx_topk, rx_topk]     # (+ tx_topk, rx_topk if desired)
    feat = np.stack(cols, axis=1).astype(np.float32)

    if zscore_cols:
        # per-feature z-score across APs (per graph), then clip again to [-3,3]
        mu = feat.mean(axis=0, keepdims=True)
        sd = feat.std(axis=0, keepdims=True) + 1e-6
        feat = (feat - mu) / sd
        feat = np.clip(feat, -clip, clip)

    return feat  # shape [A, 4] (or [A,6] if you include topk means)







def preprocess_scomm(S):
    """
    S: [A, U, U] (float32)
    Returns S_proc with per-AP standardization, symmetric, non-negative.
    """
    S = S.astype(np.float32)
    # symmetrize per-AP and take magnitude
    S = 0.5*(S + np.transpose(S, (0,2,1)))
    S = np.abs(S)
    # per-AP z-score to stabilize scales
    A, U, _ = S.shape
    Sf = S.reshape(A, -1)
    mean = Sf.mean(axis=1, keepdims=True)
    std  = Sf.std(axis=1, keepdims=True) + 1e-6
    Sn = ((Sf - mean)/std).reshape(A, U, U)
    # clamp to avoid crazy outliers
    return np.clip(Sn, 0.0, 6.0).astype(np.float32)





def per_ap_rank01(Gn):
    # 1.0 = strongest per AP
    A, U = Gn.shape
    ranks = np.argsort(np.argsort(-Gn, axis=1), axis=1)  # 0 strongest
    return 1.0 - ranks/(U-1+1e-9)





def make_edge_features(G_comm, S_comm, topk=3):
    """
    Inputs:
      G_comm [A,U] linear -> we normalize inside
      S_comm [A,U,U]
    Returns:
      edge_attr [E, F] for E=A*U edges, features per (a,u) in order a-major
    """
    Gn = normalize_gcomm(G_comm)   # [A,U]
    S  = preprocess_scomm(S_comm)            # [A,U,U]
    A, U = Gn.shape

    # interference field per (a,u,v)
    I = S * Gn[:, None, :]                   # [A,U,U]
    # sum, max, topk-mean excluding self
    mask = np.ones((A, U, U), dtype=bool)
    np.fill_diagonal(mask.reshape(A, U*U), False)  # clear (u==v) per AP

    I_masked = np.where(mask, I, -np.inf)
    interf_sum = (np.where(mask, I, 0.0)).sum(axis=2)                         # [A,U]
    interf_max = np.max(I_masked, axis=2)                                     # [A,U]
    # top-k mean
    part = np.partition(I_masked, -topk, axis=2)[:, :, -topk:]              # [A,U,topk]
    interf_topk_mean = np.where(np.isfinite(part), part, 0.0).mean(axis=2)    # [A,U]
    # spread (std) on valid entries
    valid = np.where(mask, I, 0.0)
    spread = valid.std(axis=2)                                                # [A,U]

    rank01 = per_ap_rank01(Gn)                                                # [A,U]

    # stack features in this order:
    # [gain_norm, rank01, interf_sum, interf_max, interf_topk_mean, spread]
    feats = np.stack([Gn, rank01, interf_sum, interf_max, interf_topk_mean, spread], axis=2)  # [A,U,6]
    return feats.astype(np.float32)






def to_hetero_xonly_old(inst: XOnlyInstance, device="cpu") -> HeteroData:
    Gcn = normalize_gcomm(inst.G_comm, clip=3.0)
    A, U = Gcn.shape
    alpha = np.float32(inst.alpha)
    lam_u = np.asarray(inst.lambda_cu, np.float32).reshape(U)
    x_lbl = np.asarray(inst.x, np.float32)

    data = HeteroData()
    data['ap'].x   = torch.full((A,1), alpha)                   # [A,1]
    data['user'].x = torch.from_numpy(np.stack([np.full(U,alpha,np.float32), lam_u], axis=1))  # [U,2]

    a_idx, u_idx = np.nonzero(np.ones_like(Gcn, dtype=np.int32))
    ei = torch.tensor(np.stack([a_idx, u_idx], 0), dtype=torch.long)
    ea = torch.from_numpy(Gcn[a_idx, u_idx].reshape(-1,1))

    data[('ap','serves','user')].edge_index = ei
    data[('ap','serves','user')].edge_attr  = ea
    data[('ap','serves','user')].y          = torch.from_numpy(x_lbl[a_idx, u_idx].reshape(-1,1))

    return data.to(device)







def to_hetero_xonly(inst: XOnlyInstance, device="cpu") -> HeteroData:
    A, U = inst.G_comm.shape
    edge_feat = make_edge_features(inst.G_comm, inst.S_comm, topk=3)   # [A,U,6]

    a_idx, u_idx = np.nonzero(np.ones((A,U), dtype=np.int32))
    ei = torch.tensor(np.stack([a_idx, u_idx], 0), dtype=torch.long)
    ea = torch.from_numpy(edge_feat[a_idx, u_idx])                # [E,6]

    Gsn = normalize_gsens(inst.G_sens)
    ap_press = ap_sensing_features(Gsn, topk=5)     # [A,6] sensing features per AP, ~[-3,3]

    alpha_col = np.full((ap_press.shape[0], 1), np.float32(inst.alpha))
    ap_feat = np.concatenate([alpha_col, ap_press], axis=1)  # [A, 1+6]

    data = HeteroData()
    data['ap'].x   = torch.from_numpy(ap_feat)   # torch.full((A,1), np.float32(inst.alpha))
    data['user'].x = torch.from_numpy(np.stack([np.full(U,inst.alpha,np.float32),
                                                inst.lambda_cu.astype(np.float32)], axis=1))
    data[('ap','serves','user')].edge_index = ei
    data[('ap','serves','user')].edge_attr  = ea                   # << d_edge = 6
    data[('ap','serves','user')].y          = torch.from_numpy(inst.x[a_idx, u_idx].reshape(-1,1).astype(np.float32))
    return data.to(device)





from torch.utils.data import Dataset

class XOnlyDataset(Dataset):
    def __init__(self, G_comm_list, S_comm_list, G_sens_list, alpha_list, lambda_cu_list, solution_list, device="cpu"):
        self.Gc = G_comm_list
        self.Sc = S_comm_list
        self.Gs = G_sens_list
        self.al = alpha_list
        self.lu = lambda_cu_list
        self.sol = solution_list
        self.device = device

    def __len__(self): return len(self.Gc)

    def __getitem__(self, i):
        Gc = self.Gc[i]
        S = self.Sc[i]
        Gs = self.Gs[i]
        alpha = float(self.al[i])
        lam_u = self.lu[i]
        # adapt access if solution is a dataclass: e.g., self.sol[i].x
        x = self.sol[i].x
        inst = XOnlyInstance(G_comm=Gc, S_comm=S, G_sens=Gs, alpha=alpha, lambda_cu=lam_u, x=x)
        return to_hetero_xonly(inst, device=self.device)







def data_loader(nsamps):
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
                    'solution': solution_list[:nsamps]}
    return result_lists
