
# file: assent_data_v3.py
from torch_geometric.data import HeteroData
from assent_utils_v3 import *
import src.utils.library as lib
import os

def build_graph(inst: AssentData) -> HeteroData:
    """
    Creates a tri-partite HeteroData with nodes: ap, user, target and edges:
      ('ap','serves','user'), ('ap','senses_tx','target'), ('ap','senses_rx','target')
    Sparsifies AP↔Target by top-K per AP (separately for tx/rx).
    Edge dims:
      serves edge_attr: at least 1 (G_comm z-score), keep your extra features if you had them.
      senses_* edge_attr: 3 (max, lse, topk_mean) from G_sens.
    """
    Gsn = normalize_gsens(inst.G_sens)                 # [A,A,T] in ~[-3,3]
    tx_feat, rx_feat = ap_target_edge_feats(Gsn, topk=3)  # [A,T,3] each

    A, U = inst.G_comm.shape
    T    = inst.lambda_tg.shape[0]

    Gsn = normalize_gsens(inst.G_sens)
    ap_press = ap_sensing_features(Gsn, topk=5)  # [A,6] sensing features per AP, ~[-3,3]

    alpha_col = np.full((ap_press.shape[0], 1), np.float32(inst.alpha))
    ap_feat = np.concatenate([alpha_col, ap_press], axis=1)  # [A, 1+6]

    data = HeteroData()

    # ---- Nodes
    data['ap'].x   = torch.from_numpy(ap_feat)
    data['user'].x = torch.from_numpy(np.stack([np.full(U, inst.alpha, np.float32),
                                               np.asarray(inst.lambda_cu, np.float32)], axis=1))  # [U,2]
    data['target'].x = torch.from_numpy(np.stack([np.full(T, inst.alpha, np.float32),
                                                  np.asarray(inst.lambda_tg,np.float32)], axis=1))  # [T,2]

    # ---- AP↔User edges (fully connected as before, and features added by ap_user_edge_feats)
    ap_user_edge_f = ap_user_edge_feats(inst.G_comm, inst.S_comm, topk=3)
    a_idx, u_idx = np.nonzero(np.ones((A, U), dtype=np.int32))
    ei_serves = torch.tensor(np.stack([a_idx, u_idx], 0), dtype=torch.long)
    ea_serves = torch.from_numpy(ap_user_edge_f[a_idx, u_idx])  # [E,6]
    data[('ap','serves','user')].edge_index = ei_serves
    data[('ap','serves','user')].edge_attr  = ea_serves

    # ---- AP (Tx) ↔ Target edges: pick top-K targets per AP by tx_max (col 0 of tx_feat)
    tx_score = tx_feat[:,:,0]  # [A,T]
    topk = min(inst.K_tx, T)
    tx_idx = np.argpartition(-tx_score, kth=topk-1, axis=1)[:, :topk]   # [A,K_tx] unsorted
    atx = np.repeat(np.arange(A)[:,None], topk, axis=1).reshape(-1)     # [A*K_tx]
    ttx = tx_idx.reshape(-1)
    ei_tx = torch.tensor(np.stack([atx, ttx], 0), dtype=torch.long)
    ea_tx = torch.from_numpy(tx_feat[atx, ttx])  # [E_tx, 3]
    data[('ap','senses_tx','target')].edge_index = ei_tx
    data[('ap','senses_tx','target')].edge_attr  = ea_tx

    # ---- AP (Rx) ↔ Target edges: pick top-K by rx_max (col 0 of rx_feat)
    rx_score = rx_feat[:,:,0]  # [A,T]
    topk = min(inst.K_rx, T)
    rx_idx = np.argpartition(-rx_score, kth=topk-1, axis=1)[:, :topk]
    arx = np.repeat(np.arange(A)[:,None], topk, axis=1).reshape(-1)
    trx = rx_idx.reshape(-1)
    ei_rx = torch.tensor(np.stack([arx, trx], 0), dtype=torch.long)
    ea_rx = torch.from_numpy(rx_feat[arx, trx])  # [E_rx, 3]
    data[('ap','senses_rx','target')].edge_index = ei_rx
    data[('ap','senses_rx','target')].edge_attr  = ea_rx

    return data.to(inst.device)







from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, G_comm_list, S_comm_list, G_sens_list,
                 alpha_list, lambda_cu_list, lambda_tg_list, solution_list, device="cpu",
                 N_RF=4, K_tx=6, K_rx=6):
        self.Gc = G_comm_list
        self.Sc = S_comm_list
        self.Gs = G_sens_list
        self.alpha = alpha_list
        self.lcu = lambda_cu_list
        self.ltg = lambda_tg_list
        self.sol = solution_list
        self.device = device
        self.N_RF = N_RF
        self.K_tx, self.K_rx = K_tx, K_rx

    def __len__(self): return len(self.Gc)

    def __getitem__(self, i):
        inst = AssentData(G_comm=self.Gc[i], S_comm=self.Sc[i], G_sens=self.Gs[i],
                          alpha=self.alpha[i], lambda_cu=self.lcu[i], lambda_tg=self.ltg[i],
                          solution=self.sol[i], N_RF=self.N_RF, K_tx=self.K_tx, K_rx=self.K_rx, device=self.device)
        data = build_graph(inst)

        # attach x labels on serves edges
        x_lbl = np.asarray(self.sol[i].x, np.float32)  # or .x for dataclass
        a_idx, u_idx = data[('ap','serves','user')].edge_index.cpu().numpy()
        data[('ap','serves','user')].y = torch.from_numpy(x_lbl[a_idx, u_idx].reshape(-1,1))

        # tau node labels
        tau = np.asarray(self.sol[i].tau, np.float32).reshape(-1, 1)
        data['ap'].y = torch.from_numpy(tau)

        # s node labels
        s = np.asarray(self.sol[i].s, np.float32).reshape(-1, 1)
        data['target'].y = torch.from_numpy(s)
        return to_cpu_fp32(data)






def save_dataset_to_shards(ds: GraphDataset, out_dir: str, shard_size: int = 1000, seed: int = 42):
    from tqdm import tqdm
    os.makedirs(out_dir, exist_ok=True)
    N = len(ds)
    # Build once and save shards
    graphs = []
    for i in tqdm(range(N), desc="Materializing graphs"):
        graphs.append(ds[i])
        # Save a shard
        if (i+1) % shard_size == 0:
            s = i+1 - shard_size
            torch.save(graphs, os.path.join(out_dir, f"graphs_{s:05d}_{i+1:05d}.pt"))
            graphs = []
    # leftover
    if graphs:
        s = N - len(graphs)
        torch.save(graphs, os.path.join(out_dir, f"graphs_{s:05d}_{N:05d}.pt"))

    # Fixed split indices (reproducible)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    ntr, nva = int(0.8*N), int(0.1*N)
    split = {
        "train_idx": perm[:ntr].tolist(),
        "val_idx":   perm[ntr:ntr+nva].tolist(),
        "test_idx":  perm[ntr+nva:].tolist()
    }
    torch.save(split, os.path.join(out_dir, "split.pt"))
    lib.print_log(tag='SAVE', message=f"Saved split and shards to {out_dir}")





def precompute_graph_dataset(nsamps: int = 10000, save_dataset: bool = True, seed: int = 42):
    import random
    # results: the 10k list of dicts/dataclasses

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    OUT_DIR = "cache_graphs"
    os.makedirs(OUT_DIR, exist_ok=True)
    results = data_loader(nsamps=nsamps)
    ds = GraphDataset(results["G_comm"], results["S_comm"], results["G_sens"],
                      results["alpha"], results["lambda_cu"], results["lambda_tg"],
                      results["solution"], N_RF=results["N_RF"], device="cpu")

    if save_dataset: save_dataset_to_shards(ds, OUT_DIR, shard_size=1000, seed=seed)


# -------------- Main -------------- #
if __name__ == "__main__":
    precompute_graph_dataset(nsamps=10000, save_dataset=True, seed=42)
