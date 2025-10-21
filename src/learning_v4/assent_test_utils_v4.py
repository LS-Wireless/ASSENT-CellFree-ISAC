

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, brier_score_loss

@torch.no_grad()
def evaluate_rel_metrics(val_loader, model, device, rel, thr_opt: float = 0.5, average: str = "binary"):
    """
    Evaluate a single output variable corresponding to `rel`.

    Args:
        val_loader: DataLoader over HeteroData graphs (labels must be attached).
        model:      your trained model; must return dict with keys:
                    'x_logit', 'tau_logit', 's_logit', 'ytx_logit', 'yrx_logit'
        device:     torch device string (e.g., 'cuda' or 'cpu')
        rel:        relation or node type:
                      - 'ap'         -> tau
                      - 'target'     -> s
                      - ('ap','serves','user')         -> x
                      - ('ap','senses_tx','target')    -> y_tx
                      - ('ap','senses_rx','target')    -> y_rx
        thr_opt:    probability threshold for binarizing predictions
        average:    averaging for sklearn metrics ('binary', 'macro', etc.)

    Returns:
        metrics: dict with {'f1': float, 'recall': float, 'brier': float,
                            'n': int} where n is number of evaluated items.
    """
    def _key_for_rel(r):
        if isinstance(r, tuple):
            if r == ('ap','serves','user'):
                return 'x_logit'
            elif r == ('ap','senses_tx','target'):
                return 'ytx_logit'
            elif r == ('ap','senses_rx','target'):
                return 'yrx_logit'
            else:
                raise ValueError(f"Unknown edge relation: {r}")
        else:
            if r == 'ap':
                return 'tau_logit'
            elif r == 'target':
                return 's_logit'
            else:
                raise ValueError(f"Unknown node type: {r}")

    logit_key = _key_for_rel(rel)

    probs_all = []
    y_all = []

    model.eval()
    for b in val_loader:
        b = b.to(device)
        out = model(b)

        # Pull logits and labels depending on rel type
        if isinstance(rel, tuple):  # edge task
            if 'y' not in b[rel]:
                continue  # skip if labels not present
            logits = out[logit_key].view(-1)
            y_true = b[rel].y.view(-1).to(logits.dtype)
        else:  # node task
            if 'y' not in b[rel]:
                continue
            logits = out[logit_key].view(-1)
            y_true = b[rel].y.view(-1).to(logits.dtype)

        y_prob = torch.sigmoid(logits)

        probs_all.append(y_prob.detach().cpu())
        y_all.append(y_true.detach().cpu())

    if len(y_all) == 0:
        # No labels found for this relation in the loader
        return {"f1": np.nan, "recall": np.nan, "brier": np.nan, "n": 0}

    y_true = torch.cat(y_all).numpy().astype(np.int32)
    y_prob = torch.cat(probs_all).numpy().astype(np.float32)
    y_pred = (y_prob > float(thr_opt)).astype(np.int32)

    # sklearn metrics (robust to all-zeros cases with zero_division handling)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)

    # Brier score is defined for binary probabilities; use positive class prob
    # If using macro average for multi-label, we still report the global Brier on all items
    try:
        brier = brier_score_loss(y_true, y_prob, pos_label=1)
    except ValueError:
        # Fallback if something odd about labels
        brier = np.nan

    return {"f1": float(f1), "recall": float(rec), "brier": float(brier), "n": int(y_true.size)}
