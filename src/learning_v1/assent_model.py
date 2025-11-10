import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleDict, BCEWithLogitsLoss, LayerNorm
from torch_geometric.nn import SAGEConv, HeteroConv, GATv2Conv, GraphConv
from sklearn.metrics import f1_score
import numpy as np


def calculate_utility_score(predictions, data):
    # This is where we will implement MILP's objective function.
    return np.random.rand()


class ASSENT(Module):
    def __init__(self, node_feature_dims: dict, hidden_dim=64, heads=4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.heads = heads

        # Input projection layers to map diverse features to a common hidden dimension
        self.proj = ModuleDict({
            'ap': Linear(node_feature_dims['ap'], hidden_dim),
            'user': Linear(node_feature_dims['user'], hidden_dim),
            'target': Linear(node_feature_dims['target'], hidden_dim),
        })

        # The core message passing layers using HeteroConv
        self.conv1 = HeteroConv({
            ('ap', 'serves', 'user'): GATv2Conv((-1, -1), hidden_dim, heads=heads, add_self_loops=False),
            ('ap', 'senses', 'target'): GATv2Conv((-1, -1), hidden_dim, heads=heads, add_self_loops=False),
            # Reverse edges are crucial for bi-directional information flow
            ('user', 'rev_serves', 'ap'): GATv2Conv((-1, -1), hidden_dim, heads=heads, add_self_loops=False),
            ('target', 'rev_senses', 'ap'): GATv2Conv((-1, -1), hidden_dim, heads=heads, add_self_loops=False),
        }, aggr='sum')

        # The output of conv1 is hidden_dim * heads
        self.ln1 = ModuleDict({
            'ap': LayerNorm(hidden_dim * heads),
            'user': LayerNorm(hidden_dim * heads),
            'target': LayerNorm(hidden_dim * heads),
        })

        self.conv2 = HeteroConv({
            ('ap', 'serves', 'user'): GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, add_self_loops=False),
            ('ap', 'senses', 'target'): GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, add_self_loops=False),
            ('user', 'rev_serves', 'ap'): GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, add_self_loops=False),
            ('target', 'rev_senses', 'ap'): GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, add_self_loops=False),
        }, aggr='sum')

        # The output of conv2 is just hidden_dim
        self.ln2 = ModuleDict({
            'ap': LayerNorm(hidden_dim),
            'user': LayerNorm(hidden_dim),
            'target': LayerNorm(hidden_dim),
        })

        # # The third layer (optional)
        # self.conv3 = HeteroConv({
        #     ('ap', 'serves', 'user'): GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, add_self_loops=False),
        #     ('ap', 'senses', 'target'): GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, add_self_loops=False),
        #     ('user', 'rev_serves', 'ap'): GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, add_self_loops=False),
        #     ('target', 'rev_senses', 'ap'): GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, add_self_loops=False),
        # }, aggr='sum')
        # # ---------------

        # # --- TEMPORARY: Use a single, simple GCNConv layer ---
        # self.conv1 = HeteroConv({
        #     ('ap', 'serves', 'user'): GraphConv(-1, hidden_dim),
        #     ('ap', 'senses', 'target'): GraphConv(-1, hidden_dim),
        #     ('user', 'rev_serves', 'ap'): GraphConv(-1, hidden_dim),
        #     ('target', 'rev_senses', 'ap'): GraphConv(-1, hidden_dim),
        # }, aggr='mean')  # Use 'mean' aggregation for GCN, it's more stable

        final_embedding_dim = hidden_dim * heads
        final_embedding_dim = hidden_dim

        # --- Define Prediction Heads for each output variable ---
        self.ap_mode_head = Linear(final_embedding_dim, 1)  # Predicts tau
        self.target_sched_head = Linear(final_embedding_dim, 1)  # Predicts s
        self.user_assoc_head = Linear(2 * final_embedding_dim, 1)  # Predicts x
        self.sens_tx_head = Linear(2 * final_embedding_dim, 1)  # Predicts y_tx
        self.sens_rx_head = Linear(2 * final_embedding_dim, 1)  # Predicts y_rx

    def forward(self, data):
        # Apply initial linear projections
        x_dict = {key: self.proj[key](data[key].x) for key in data.node_types}

        # First round of message passing
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: self.ln1[key](x) for key, x in x_dict.items()}
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        # Second round of message passing
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {key: self.ln2[key](x) for key, x in x_dict.items()}
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        # # Third round of message passing
        # x_dict = self.conv3(x_dict, data.edge_index_dict)
        # x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        # # -------------------------------------------

        # # --- TEMPORARY: Use only the one GCN layer ---
        # x_dict = self.conv1(x_dict, data.edge_index_dict)
        # x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        # --- Generate Final Predictions ---
        # 1. Node-level predictions (tau, s)
        tau_logits = self.ap_mode_head(x_dict['ap'])
        s_logits = self.target_sched_head(x_dict['target'])

        # 2. Edge-level predictions (x, y_tx, y_rx)
        # For user association 'x'
        ap_user_edge_index = data['ap', 'serves', 'user'].edge_index
        ap_emb = x_dict['ap'][ap_user_edge_index[0]]
        user_emb = x_dict['user'][ap_user_edge_index[1]]
        x_logits = self.user_assoc_head(torch.cat([ap_emb, user_emb], dim=-1))

        # For sensing associations 'y_tx' and 'y_rx'
        ap_target_edge_index = data['ap', 'senses', 'target'].edge_index
        ap_emb_sens = x_dict['ap'][ap_target_edge_index[0]]
        target_emb = x_dict['target'][ap_target_edge_index[1]]
        sens_concat_emb = torch.cat([ap_emb_sens, target_emb], dim=-1)
        ytx_logits = self.sens_tx_head(sens_concat_emb)
        yrx_logits = self.sens_rx_head(sens_concat_emb)

        return {
            'tau': tau_logits.squeeze(-1), 's': s_logits.squeeze(-1), 'x': x_logits.squeeze(-1),
            'ytx': ytx_logits.squeeze(-1), 'yrx': yrx_logits.squeeze(-1)
        }


def train_epoch(model, loader, optimizer, loss_fns: dict, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)

        # Calculate loss for ONLY the 'tau' prediction head
        # loss = loss_fns['tau'](out['tau'], batch['ap'].y_tau)

        # Sum the losses from all prediction heads
        loss = (loss_fns['tau'](out['tau'], batch['ap'].y_tau)
                + loss_fns['s'](out['s'], batch['target'].y_s)
                + loss_fns['x'](out['x'], batch['ap', 'serves', 'user'].y_x)
                + loss_fns['ytx'](out['ytx'], batch['ap', 'senses', 'target'].y_ytx)
                + loss_fns['yrx'](out['yrx'], batch['ap', 'senses', 'target'].y_yrx))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, device, milp_utilities=None):
    model.eval()
    f1_scores = []
    utility_ratios = []

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        out = model(batch)

        # Convert logits to binary predictions
        preds = {key: (torch.sigmoid(val) > 0.5).int() for key, val in out.items()}

        # --- Metric 1: F1-Score (how well we mimic the MILP) ---
        # We calculate F1 for user association 'x' as an example
        y_true_x = batch['ap', 'serves', 'user'].y_x.cpu().numpy()
        y_pred_x = preds['x'].cpu().numpy()
        # f1_scores.append(f1_score(y_true_x, y_pred_x, zero_division=0))

        y_true_tau = batch['ap'].y_tau.cpu().numpy()
        y_pred_tau = preds['tau'].cpu().numpy()
        f1_scores.append(f1_score(y_true_tau, y_pred_tau, zero_division=0))

        if milp_utilities is not None:
            # --- Metric 2: Utility Score (the most important metric) ---
            # Calculate the utility achieved by the GNN's decisions and compare to the MILP's optimal utility
            start_idx = i * loader.batch_size
            end_idx = start_idx + batch.num_graphs
            batch_milp_utilities = milp_utilities[start_idx:end_idx]

            # The 'calculate_utility_score' function is implementation of the MILP objective
            # It needs the GNN's decisions and the original channel gains from the batch data
            gnn_utility = calculate_utility_score(preds, batch)

            # Avoid division by zero if MILP utility was 0
            ratio = gnn_utility / (batch_milp_utilities.mean() + 1e-9)
            utility_ratios.append(ratio)
    if milp_utilities is None:
        return np.mean(f1_scores)
    else:
        return np.mean(f1_scores), np.mean(utility_ratios)
