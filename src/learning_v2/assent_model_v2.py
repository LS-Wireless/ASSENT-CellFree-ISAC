# file: x_only_model.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import HeteroConv, NNConv

def mlp(din, dh, dout, drop=0.1):
    return nn.Sequential(nn.Linear(din,dh), nn.ReLU(), nn.Dropout(drop), nn.Linear(dh,dout))

class XOnlyGNN(nn.Module):
    def __init__(self, d_ap=7, d_user=2, d_edge=6, hidden=64, layers=2, drop=0.1):
        super().__init__()
        self.ap_in   = mlp(d_ap,   hidden, hidden, drop)
        self.user_in = mlp(d_user, hidden, hidden, drop)

        self.layers = nn.ModuleList()
        for _ in range(layers):
            net = mlp(d_edge, hidden, hidden*hidden, drop)  # for NNConv weights
            self.layers.append(HeteroConv({('ap','serves','user'): NNConv(hidden, hidden, net, aggr='mean'),
                                           ('user','rev_served_by','ap'): NNConv(hidden, hidden, net, aggr='mean')},
                                          aggr='sum'))
        self.edge_head = mlp(2*hidden + d_edge, hidden, 1, drop)
        self.ap_head = mlp(hidden, hidden, 1, drop)

    def forward(self, data):
        x = {'ap': self.ap_in(data['ap'].x),
             'user': self.user_in(data['user'].x)}
        # build both directions’ edge_index/attr
        ei_fwd = data[('ap','serves','user')].edge_index
        ea     = data[('ap','serves','user')].edge_attr
        ei_rev = torch.flip(ei_fwd, [0])
        edge_index_dict = {('ap','serves','user'): ei_fwd, ('user','rev_served_by','ap'): ei_rev}
        edge_attr_dict  = {('ap','serves','user'): ea,     ('user','rev_served_by','ap'): ea}

        for conv in self.layers:
            x = conv(x, edge_index_dict, edge_attr_dict)
            x = {k: F.relu(v) for k,v in x.items()}

        # edge logits for x
        h_ap   = x['ap'][ei_fwd[0]]
        h_user = x['user'][ei_fwd[1]]
        feats  = torch.cat([h_ap, h_user, ea], dim=-1)
        x_logit = self.edge_head(feats)    # [E,1]

        # node logits for tau
        tau_logit = self.ap_head(x['ap'])  # [A,1]
        return {'x_logit': x_logit, 'tau_logit': tau_logit}

