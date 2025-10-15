import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleDict, BCEWithLogitsLoss
from torch_geometric.nn import SAGEConv, HeteroConv
from sklearn.metrics import f1_score
import numpy as np


# A placeholder for your utility calculation logic
def calculate_utility_score(predictions, data):
    # --- IMPORTANT ---
    # This is where you will implement your MILP's objective function.
    # You'll take the GNN's predicted decisions (e.g., predictions['x'])
    # and the original channel gains from the 'data' object to calculate
    # the total utility achieved by the GNN's policy.
    # For this example, we'll just return a dummy value.
    return np.random.rand()


class ASSENT(Module):
    def __init__(self, node_feature_dims: dict, hidden_dim=64):
        super().__init__()

        # Input projection layers to map diverse features to a common hidden dimension
        self.proj = ModuleDict({
            'ap': Linear(node_feature_dims['ap'], hidden_dim),
            'user': Linear(node_feature_dims['user'], hidden_dim),
            'target': Linear(node_feature_dims['target'], hidden_dim),
        })

        # The core message passing layer using HeteroConv
        self.conv1 = HeteroConv({
            ('ap', 'serves', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('ap', 'senses', 'target'): SAGEConv((-1, -1), hidden_dim),
            # Reverse edges are crucial for bi-directional information flow
            ('user', 'rev_serves', 'ap'): SAGEConv((-1, -1), hidden_dim),
            ('target', 'rev_senses', 'ap'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('ap', 'serves', 'user'): SAGEConv(hidden_dim, hidden_dim),
            ('ap', 'senses', 'target'): SAGEConv(hidden_dim, hidden_dim),
            ('user', 'rev_serves', 'ap'): SAGEConv(hidden_dim, hidden_dim),
            ('target', 'rev_senses', 'ap'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')

        # --- Define Prediction Heads for each output variable ---
        self.ap_mode_head = Linear(hidden_dim, 1)  # Predicts tau
        self.target_sched_head = Linear(hidden_dim, 1)  # Predicts s
        self.user_assoc_head = Linear(2 * hidden_dim, 1)  # Predicts x
        self.sens_tx_head = Linear(2 * hidden_dim, 1)  # Predicts y_tx
        self.sens_rx_head = Linear(2 * hidden_dim, 1)  # Predicts y_rx

    def forward(self, data):
        # Apply initial linear projections
        x_dict = {key: self.proj[key](data[key].x) for key in data.node_types}

        # First round of message passing
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        # Second round of message passing
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

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

        # Sum the losses from all prediction heads
        # loss = (loss_fn(out['tau'], batch['ap'].y_tau)
        #         + loss_fn(out['s'], batch['target'].y_s)
        #         + loss_fn(out['x'], batch['ap', 'serves', 'user'].y_x)
        #         + loss_fn(out['ytx'], batch['ap', 'senses', 'target'].y_ytx)
        #         + loss_fn(out['yrx'], batch['ap', 'senses', 'target'].y_yrx))
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
        y_true = batch['ap', 'serves', 'user'].y_x.cpu().numpy()
        y_pred = preds['x'].cpu().numpy()
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        if milp_utilities is not None:
            # --- Metric 2: Utility Score (the most important metric) ---
            # Calculate the utility achieved by the GNN's decisions and compare to the MILP's optimal utility
            start_idx = i * loader.batch_size
            end_idx = start_idx + batch.num_graphs
            batch_milp_utilities = milp_utilities[start_idx:end_idx]

            # The 'calculate_utility_score' function is your implementation of the MILP objective
            # It needs the GNN's decisions and the original channel gains from the batch data
            gnn_utility = calculate_utility_score(preds, batch)

            # Avoid division by zero if MILP utility was 0
            ratio = gnn_utility / (batch_milp_utilities.mean() + 1e-9)
            utility_ratios.append(ratio)
    if milp_utilities is None:
        return np.mean(f1_scores)
    else:
        return np.mean(f1_scores), np.mean(utility_ratios)
