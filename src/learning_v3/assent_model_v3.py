
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import HeteroConv, NNConv

def mlp(din, dh, dout, drop=0.1):
    return nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Dropout(drop), nn.Linear(dh, dout))

class ASSENTGNN(nn.Module):
    def __init__(self, d_ap=7, d_user=2, d_target=2,
                 d_edge_serves=6, d_edge_sens=3,
                 hidden=96, layers=3, drop=0.2, use_layernorm=True):
        super().__init__()
        # Input encoders
        self.ap_in      = mlp(d_ap,     hidden, hidden, drop)
        self.user_in    = mlp(d_user,   hidden, hidden, drop)
        self.target_in  = mlp(d_target, hidden, hidden, drop)

        self.ln_ap = nn.LayerNorm(hidden) if use_layernorm else nn.Identity()
        self.ln_user = nn.LayerNorm(hidden) if use_layernorm else nn.Identity()
        self.ln_tgt = nn.LayerNorm(hidden) if use_layernorm else nn.Identity()

        # HeteroConv message passing layers
        self.layers = nn.ModuleList()
        for _ in range(layers):
            net_serv = mlp(d_edge_serves, hidden, hidden*hidden, drop)
            net_sens = mlp(d_edge_sens,   hidden, hidden*hidden, drop)
            self.layers.append(HeteroConv({
                ('ap','serves','user'):          NNConv(hidden, hidden, net_serv, aggr='mean'),
                ('user','rev_served_by','ap'):   NNConv(hidden, hidden, net_serv, aggr='mean'),

                ('ap','senses_tx','target'):     NNConv(hidden, hidden, net_sens, aggr='mean'),
                ('target','rev_sensed_tx','ap'): NNConv(hidden, hidden, net_sens, aggr='mean'),

                ('ap','senses_rx','target'):     NNConv(hidden, hidden, net_sens, aggr='mean'),
                ('target','rev_sensed_rx','ap'): NNConv(hidden, hidden, net_sens, aggr='mean'),
            }, aggr='sum'))
        # Heads
        # 1) Edge head for x on ('ap','serves','user'): concat h_ap, h_user, serves edge_attr
        self.edge_head = mlp(2 * hidden + d_edge_serves, hidden, 1, drop)
        # 2) AP node head for tau
        self.ap_head = mlp(hidden, hidden, 1, drop)
        # 3) Target node head for s
        self.tgt_head = mlp(hidden, hidden, 1, drop)

        # learnable task uncertainty (log variances)
        self.log_var_x = torch.nn.Parameter(torch.tensor(0.0))  # for edge task x
        self.log_var_tau = torch.nn.Parameter(torch.tensor(0.0))  # for node task tau
        self.log_var_s = torch.nn.Parameter(torch.tensor(0.0))  # for node task s

    def forward(self, data):
        # Encode node features
        h = {
            'ap':     self.ln_ap(self.ap_in(data['ap'].x)),
            'user':   self.ln_user(self.user_in(data['user'].x)),
            'target': self.ln_tgt(self.target_in(data['target'].x)),
        }

        # Grab edge indices/attrs
        ei_s   = data[('ap','serves','user')].edge_index
        ea_s   = data[('ap','serves','user')].edge_attr
        ei_s_r = torch.flip(ei_s, [0])

        ei_tx   = data[('ap','senses_tx','target')].edge_index
        ea_tx   = data[('ap','senses_tx','target')].edge_attr
        ei_tx_r = torch.flip(ei_tx, [0])

        ei_rx   = data[('ap','senses_rx','target')].edge_index
        ea_rx   = data[('ap','senses_rx','target')].edge_attr
        ei_rx_r = torch.flip(ei_rx, [0])

        # Message passing
        for conv in self.layers:
            h = conv(
                h,
                edge_index_dict={
                    ('ap','serves','user'):            ei_s,
                    ('user','rev_served_by','ap'):     ei_s_r,
                    ('ap','senses_tx','target'):       ei_tx,
                    ('target','rev_sensed_tx','ap'):   ei_tx_r,
                    ('ap','senses_rx','target'):       ei_rx,
                    ('target','rev_sensed_rx','ap'):   ei_rx_r,
                },
                edge_attr_dict={
                    ('ap','serves','user'):            ea_s,
                    ('user','rev_served_by','ap'):     ea_s,
                    ('ap','senses_tx','target'):       ea_tx,
                    ('target','rev_sensed_tx','ap'):   ea_tx,
                    ('ap','senses_rx','target'):       ea_rx,
                    ('target','rev_sensed_rx','ap'):   ea_rx,
                }
            )
            # nonlinearity
            h = {k: F.relu(v) for k,v in h.items()}

        # ---- Heads ----
        # x (edge logits on serves)
        h_ap   = h['ap'][ei_s[0]]     # [E, hidden]
        h_user = h['user'][ei_s[1]]   # [E, hidden]
        x_feats = torch.cat([h_ap, h_user, ea_s], dim=-1)   # [E, 2*hidden + d_edge_serves]
        x_logit = self.edge_head(x_feats)                   # [E, 1]

        # tau (AP node logits)
        tau_logit = self.ap_head(h['ap'])                   # [A, 1]

        # s (Target node logits)
        s_logit = self.tgt_head(h['target'])                # [T, 1]

        return {'x_logit': x_logit, 'tau_logit': tau_logit, 's_logit': s_logit}

