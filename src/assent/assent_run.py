

from src.learning_v4.assent_run_utils_v4 import *
import src.utils.library as lib
import time
import os


# -------------------------------------------------------------------
# Training / evaluation
# -------------------------------------------------------------------
def run_training(model, train_loader, val_loader, cfg: TrainConfig,
                 save_model_state: bool = False, save_path: str = "checkpoints", warmup_epoch_save=10):
    device = cfg.device
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
    warmup_epochs = cfg.warmup_epochs

    lib.print_log(tag='RUN', message=f'Training {model.__class__.__name__} on {cfg.device} for {cfg.epochs} epochs')

    # precompute weights
    posw_x = compute_pos_weight_x(train_loader, device)
    tau_w0w1 = class_weights_node(train_loader, 'ap', device)
    s_w0w1 = class_weights_node(train_loader, 'target', device)
    posw_ytx = compute_pos_weight_edge(train_loader, ('ap', 'senses_tx', 'target'), device)
    posw_yrx = compute_pos_weight_edge(train_loader, ('ap', 'senses_rx', 'target'), device)

    from torch import amp
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    scaler = amp.GradScaler(device_type, enabled=(cfg.amp and device_type == 'cuda'))

    # --- Track best metrics for all outputs ---
    best = {
        'epoch': -1, 'state': None, "pareto": [],
        'f1x': -1.0, 'f1tau': -1.0, 'f1s': -1.0, 'f1ytx': -1.0, 'f1yrx': -1.0,
        'thr_x': 0.5, 'thr_tau': 0.5, 'thr_s': 0.5, 'thr_ytx': 0.5, 'thr_yrx': 0.5
    }

    stopper = EarlyStopper(mode=cfg.es_mode, patience=cfg.es_patience, min_delta=cfg.es_min_delta)

    # before training loop
    history = {
        "epoch": [], "lr": [],
        # train losses
        "train_loss_total": [], "train_loss_multi": [], "train_loss_linear": [],
        "train_loss_x": [], "train_loss_tau": [], "train_loss_s": [],
        "train_loss_ytx": [], "train_loss_yrx": [],
        # val loss (same objective as train for apples-to-apples)
        "val_loss_total": [], "val_loss_x": [], "val_loss_tau": [],
        "val_loss_s": [], "val_loss_ytx": [], "val_loss_yrx": [],
        # val metrics (F1s and thresholds)
        "val_f1x": [], "val_f1tau": [], "val_f1s": [], "val_f1ytx": [], "val_f1yrx": [],
        "thr_x": [], "thr_tau": [], "thr_s": [], "thr_ytx": [], "thr_yrx": []
    }
    lib.print_log(tag='RUN', message='Starting training epochs...')
    for ep in range(1, cfg.epochs + 1):
        train_start = time.time()
        # ----------------- train -----------------
        model.train()
        tr_losses = {'x': 0.0, 'tau': 0.0, 's': 0.0, 'ytx': 0.0, 'yrx': 0.0,
                     'loss_total': 0.0, 'loss_linear': 0.0, 'loss_multi': 0.0}
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with (amp.autocast(device_type=device_type, enabled=cfg.amp)):
                out = model(batch)

                # x + rf_penalty
                Lx, Lrf = loss_x(batch, out, posw_x, cfg)
                # tau / s
                Ltau = loss_tau(batch, out, tau_w0w1)
                Ls = loss_s(batch, out, s_w0w1)
                # y_tx / y_rx
                Lytx = loss_edge_bce(batch, out['ytx_logit'], ('ap', 'senses_tx', 'target'), pos_weight=posw_ytx,
                                     eps=cfg.eps_smooth)
                Lyrx = loss_edge_bce(batch, out['yrx_logit'], ('ap', 'senses_rx', 'target'), pos_weight=posw_yrx,
                                     eps=cfg.eps_smooth)

                # coupling (requires tau logits)
                if 'tau_logit' in out:
                    Lcoup = tau_coupling_penalty(batch, out['x_logit'], out['tau_logit'])
                else:
                    Lcoup = torch.tensor(0.0, device=device)

                # Soft constraints
                s_ref = get_s_ref(batch, out['s_logit'])
                pen_cap_tx = cap_penalty(batch, out['ytx_logit'], ('ap', 'senses_tx', 'target'), K_cap=cfg.K_tx_milp,
                                         s_ref=s_ref)
                pen_cap_rx = cap_penalty(batch, out['yrx_logit'], ('ap', 'senses_rx', 'target'), K_cap=cfg.K_rx_milp,
                                         s_ref=s_ref)
                pen_tx_tau, pen_rx_tau = tau_match_penalty(batch, out['ytx_logit'], out['yrx_logit'], out['tau_logit'])
                pen_gate_tx = s_gate_penalty(batch, out['ytx_logit'], ('ap', 'senses_tx', 'target'), s_ref)
                pen_gate_rx = s_gate_penalty(batch, out['yrx_logit'], ('ap', 'senses_rx', 'target'), s_ref)

                # light clamp each step before loss usage
                model.log_var_x.data.clamp_(-5.0, 3.0)
                model.log_var_tau.data.clamp_(-5.0, 3.0)
                model.log_var_s.data.clamp_(-5.0, 3.0)
                model.log_var_ytx.data.clamp_(-5.0, 3.0)
                model.log_var_yrx.data.clamp_(-5.0, 3.0)
                # Uncertainty-weighted sum:
                loss_multi = (
                        torch.exp(-model.log_var_x)     * Lx                    + model.log_var_x   +
                        torch.exp(-model.log_var_tau)   * (cfg.w_Ltau * Ltau)   + model.log_var_tau +
                        torch.exp(-model.log_var_s)     * (cfg.w_Ls * Ls)       + model.log_var_s   +
                        torch.exp(-model.log_var_ytx)   * (cfg.w_Lytx * Lytx)   + model.log_var_ytx +
                        torch.exp(-model.log_var_yrx)   * (cfg.w_Lyrx * Lyrx)   + model.log_var_yrx
                )
                loss_linear = (Lx + (cfg.w_Ltau * Ltau) + (cfg.w_Ls * Ls) + (cfg.w_Lytx * Lytx) + (cfg.w_Lyrx * Lyrx)
                               + (cfg.w_Lrf * Lrf) + (cfg.w_Lcoup * Lcoup))
                # Then add your structural penalties with fixed weights:
                loss = (loss_multi + (cfg.w_Lrf * Lrf) + (cfg.w_Lcoup * Lcoup)
                        + cfg.w_cap * (pen_cap_tx + pen_cap_rx)
                        + cfg.w_tau_match * (pen_tx_tau + pen_rx_tau)
                        + cfg.w_s_gate * (pen_gate_tx + pen_gate_rx))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            tr_losses['x'] += Lx.item()
            tr_losses['tau'] += float(Ltau.item()) if torch.is_tensor(Ltau) else 0.0
            tr_losses['s'] += float(Ls.item()) if torch.is_tensor(Ls) else 0.0
            tr_losses['ytx'] += Lytx.item()
            tr_losses['yrx'] += Lyrx.item()
            tr_losses['loss_multi'] += loss_multi.item()
            tr_losses['loss_linear'] += loss_linear.item()
            tr_losses['loss_total'] += loss.item()
        current_lr = opt.param_groups[0]['lr']
        train_time = time.time() - train_start
        ntr = max(1, len(train_loader))
        print(f"\nEpoch {ep:03d} | train-loss={tr_losses['loss_total'] / ntr:.3f} | "
              f"x-loss={tr_losses['x'] / ntr:.3f} | tau-loss={tr_losses['tau'] / ntr:.3f} | "
              f"s-loss={tr_losses['s'] / ntr:.3f} | ytx-loss={tr_losses['ytx'] / ntr:.3f} | "
              f"yrx-loss={tr_losses['yrx'] / ntr:.3f} | lr={current_lr:.2e} | train-time={train_time:.2f}s")

        # ----------------- validation (threshold search) -----------------
        eval_start = time.time()
        model.eval()
        thr_x, best_f1x, (Pu, Ru) = best_threshold_utility_f1_x(val_loader, model, device, cfg.gain_col_x,
                                                                cfg.thresh_min, cfg.thresh_max, cfg.thresh_steps)
        thr_tau, best_f1tau = best_threshold_macro_f1_nodes(val_loader, model, device, ntype='ap',
                                                            t_min=0.1, t_max=0.9, steps=33)
        thr_s, best_f1s = best_threshold_macro_f1_nodes(val_loader, model, device, ntype='target',
                                                        t_min=0.1, t_max=0.9, steps=33)
        thr_ytx, f1ytx = best_threshold_macro_f1_edge(val_loader, model, device, ('ap', 'senses_tx', 'target'))
        thr_yrx, f1yrx = best_threshold_macro_f1_edge(val_loader, model, device, ('ap', 'senses_rx', 'target'))

        val_Ltot, val_Lx, val_Ltau, val_Ls, val_Lytx, val_Lyrx = eval_epoch(model, val_loader, cfg, posw_x, tau_w0w1,
                                                                            s_w0w1, posw_ytx, posw_yrx, cfg.device)

        eval_time = time.time() - eval_start
        print(f" Validate | val-loss={val_Ltot:.4f} | x-F1={best_f1x:.3f} @ {thr_x:.2f} | tau-F1={best_f1tau:.3f} @ {thr_tau:.2f} | "
              f"s-F1={best_f1s:.3f} @ {thr_s:.2f} | ytx-F1={f1ytx:.3f} @ {thr_ytx:.2f} | yrx-F1={f1yrx:.3f} @ {thr_yrx:.2f} | "
              f"eval-time={eval_time:.2f}s")

        cand = {
            "epoch": ep,
            "f1x": best_f1x, "f1tau": best_f1tau, "f1s": best_f1s,
            "f1ytx": f1ytx, "f1yrx": f1yrx,
            "thr_x": thr_x, "thr_tau": thr_tau, "thr_s": thr_s,
            "thr_ytx": thr_ytx, "thr_yrx": thr_yrx
        }

        best["pareto"], added = update_pareto(best["pareto"], cand, max_keep=cfg.pareto_max_keep)

        if added:
            if save_model_state and ep > warmup_epoch_save:
                # save a checkpoint for this Pareto point
                save_path_pareto = os.path.join(save_dir, "pareto")
                os.makedirs(save_path_pareto, exist_ok=True)
                ckpt_path = os.path.join(save_path_pareto, f"pareto_ep{ep:03d}.pt")
                torch.save(model.state_dict(), ckpt_path)
                lib.print_log(tag='SAVE', message=f'Saved pareto checkpoint to {ckpt_path} at epoch {ep}')
            best.update({
                "epoch": ep, "state": model.state_dict(),
                "f1x": best_f1x, "f1tau": best_f1tau, "f1s": best_f1s,
                "f1ytx": f1ytx, "f1yrx": f1yrx,
                "thr_x": thr_x, "thr_tau": thr_tau, "thr_s": thr_s,
                "thr_ytx": thr_ytx, "thr_yrx": thr_yrx
            })

        # NEW: pick early-stop metric value
        if cfg.es_metric == "val_loss_total":
            es_value = val_Ltot
            es_mode = "min"
        elif cfg.es_metric == "val_loss_total_raw":
            es_value = val_Ltot  # same here if you're already computing raw; otherwise compute raw variant
            es_mode = "min"
        elif cfg.es_metric == "f1_combo":
            es_value = f1_combined(best_f1x, best_f1tau, best_f1s)
            es_mode = "max"
        else:
            # fallback to val loss
            es_value = val_Ltot
            es_mode = "min"

        # ensure stopper mode aligns
        stopper.mode = es_mode

        # update stopper
        improved = stopper.update(es_value)
        if improved:
            lib.print_log(tag='TRAIN', message=f"EarlyStop monitor improved: {cfg.es_metric}={es_value:.4f}")
        else:
            lib.print_log(tag='TRAIN', message=f"No improvement on {cfg.es_metric} for {stopper.bad_epochs}/{cfg.es_patience} epochs")

        # check stop condition
        if stopper.should_stop():
            lib.print_log(tag='TRAIN', message=f"Early stopping triggered on {cfg.es_metric}.")
            break
        # --- History ---
        history["epoch"].append(ep)
        history["lr"].append(current_lr)
        history["train_loss_total"].append(float(tr_losses['loss_total'] / ntr))
        history["train_loss_multi"].append(float(tr_losses['loss_multi'] / ntr))
        history["train_loss_linear"].append(float(tr_losses['loss_linear'] / ntr))
        history["train_loss_x"].append(float(tr_losses['x'] / ntr))
        history["train_loss_tau"].append(float(tr_losses['tau'] / ntr))
        history["train_loss_s"].append(float(tr_losses['s'] / ntr))
        history["train_loss_ytx"].append(float(tr_losses['ytx'] / ntr))
        history["train_loss_yrx"].append(float(tr_losses['yrx'] / ntr))
        history["val_loss_total"].append(float(val_Ltot))
        history["val_loss_x"].append(float(val_Lx))
        history["val_loss_tau"].append(float(val_Ltau))
        history["val_loss_s"].append(float(val_Ls))
        history["val_loss_ytx"].append(float(val_Lytx))
        history["val_loss_yrx"].append(float(val_Lyrx))
        history["val_f1x"].append(float(best_f1x))
        history["val_f1tau"].append(float(best_f1tau))
        history["val_f1s"].append(float(best_f1s))
        history["val_f1ytx"].append(float(f1ytx))
        history["val_f1yrx"].append(float(f1yrx))
        history["thr_x"].append(float(thr_x))
        history["thr_tau"].append(float(thr_tau))
        history["thr_s"].append(float(thr_s))
        history["thr_ytx"].append(float(thr_ytx))
        history["thr_yrx"].append(float(thr_yrx))

        # step LR scheduler
        if ep <= warmup_epochs:
            for g in opt.param_groups:
                g['lr'] = cfg.lr_min + (cfg.lr - cfg.lr_min) * ep / warmup_epochs
        else:
            sched.step()

    lib.print_log(tag='RUN', message='Finished training!\n')
    # load best & return summary
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    summary = {'best_f1x': best['f1x'], 'best_f1tau': best['f1tau'], 'best_f1s': best['f1s'],
               'best_f1ytx': best['f1ytx'], 'best_f1yrx': best['f1yrx'],
               'thr_x': best['thr_x'], 'thr_tau': best['thr_tau'], 'thr_s': best['thr_s'],
               'thr_ytx': best['thr_ytx'], 'thr_yrx': best['thr_yrx'],
               'pareto': best['pareto']
               }
    return model, summary, history


# -------------------------------------------------------------------
# Main function to run the experiment
# -------------------------------------------------------------------
if __name__ == "__main__":

    from src.learning_v4.assent_data_v4 import GraphDataset
    from src.learning_v4.assent_data_utils_v4 import input_data_loader
    from torch_geometric.loader import DataLoader
    from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer
    import json
    from datetime import datetime

    run_id = 'run_03a'
    load_dir = "checkpoints"
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M")
    log_path = os.path.join(load_dir, run_id, f"{timestamp}_log_{run_id}.txt")
    sys.stdout = TeeLogger(log_path)

    run_start_time = time.time()
    width = 50
    print("=" * width)
    print(">> ASSENT Run Started <<".center(width))
    print(f'{now.strftime("%A, %B %d, %Y at %I:%M:%S %p")}')
    print("=" * width)

    # Load metadata
    lib.print_log(tag='CONFIG', message=f"Run ID: '{run_id}'")
    metadata_path = os.path.join(load_dir, run_id, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    lib.print_log(tag='LOAD', message=f"Loaded metadata from '{metadata_path}'")

    config = metadata['config']
    # Check
    if config['run_id'] != run_id:
        raise ValueError(
            f"Run ID in metadata.json ({config['run_id']}) does not match run ID in function call ({run_id}).")
    lib.print_log(tag='CONFIG', message=f"SAVE_MODEL is {config['SAVE_MODEL']}")
    save_dir = os.path.join(config['save_dir'], run_id)
    lib.print_log(tag='CONFIG', message=f"save_dir is '{save_dir}'")

    train_cfg_dict = metadata['TrainConfig']
    train_cfg = TrainConfig(**train_cfg_dict)
    train_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    results = input_data_loader(path_to_input_data=config['path_to_input_data'], input_file_name=config['input_file_name'],
                                path_to_solution_data=config['path_to_solution_data'], solution_file_name=config['solution_file_name'],
                                input_parts_to_load=config['input_parts_to_load'], solution_parts_to_load=config['solution_parts_to_load'],
                                nsamps=config['nsamps'])
    train_cfg.rf_cap = results["N_RF"]
    train_cfg.K_tx_milp = results["K_tx_milp"]
    train_cfg.K_rx_milp = results["K_rx_milp"]


    # split
    N = config['nsamps']
    idx = np.random.RandomState(config['split_seed']).permutation(N)
    ntr = int(0.8 * N)
    nva = int(0.1 * N)
    tr_idx, va_idx = idx[:ntr], idx[ntr:ntr + nva]
    te_idx = idx[ntr + nva:]

    ds = GraphDataset(results["G_comm"], results["S_comm"], results["G_sens"],
                      results["alpha"], results["lambda_cu"], results["lambda_tg"],
                      results["solution"], N_RF=results["N_RF"], device=train_cfg.device,
                      K_tx_milp=results["K_tx_milp"], K_rx_milp=results["K_rx_milp"])

    from torch.utils.data import Subset

    tr_set, va_set, te_set = Subset(ds, tr_idx.tolist()), Subset(ds, va_idx.tolist()), Subset(ds, te_idx.tolist())

    lib.print_log(tag='RUN',
                  message=f"Training set size: {len(tr_set)}; Validation set size: {len(va_set)}; Test set size: {len(te_set)}")
    tr_loader = DataLoader(tr_set, batch_size=train_cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=train_cfg.batch_size, shuffle=False)
    te_loader = DataLoader(te_set, batch_size=train_cfg.batch_size, shuffle=False)

    backbone = config['backbone'] if config['backbone'] in config['_backbone_list'] else "nnconv"
    lib.print_log(tag='CONFIG', message=f"Backbone (layer type) is '{backbone}'")
    if backbone == 'gatv2':
        model = ASSENTGNN_GAT(hidden=config['nhidden'], layers=config['nlayers'], drop=config['drop'],
                              heads=config['gatv2_heads']).to(train_cfg.device)
    elif backbone == 'transformer':
        model = ASSENTGNN_Transformer(hidden=config['nhidden'], layers=config['nlayers'], drop=config['drop'],
                                      heads=config['transformer_heads'], beta=config['transformer_beta']).to(train_cfg.device)
    else:
        model = ASSENTGNN(hidden=config['nhidden'], layers=config['nlayers'], drop=config['drop']).to(train_cfg.device)
    summary = history = {}

    lib.print_log(tag='CONFIG', message=f"TRAIN is {config['TRAIN']}")
    if config['TRAIN']:
        model, summary, history = run_training(model, tr_loader, va_loader, train_cfg,
                                               save_model_state=config['save_pareto'],
                                               save_path=save_dir, warmup_epoch_save=config['pareto_warmup_epochs'])


    if config['SAVE_MODEL']:

        # === Create timestamped save directory ===
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        os.makedirs(save_dir, exist_ok=True)

        # === Save model weights ===
        model_path = os.path.join(save_dir, timestamp+"_best_model_fully_trained.pt")
        torch.save(model.state_dict(), model_path)
        lib.print_log(tag='SAVE', message=f"Saved model weights to '{model_path}'")

        # --- Save Summary as JSON ---
        hist_json = os.path.join(save_dir, timestamp+"_history.json")
        with open(hist_json, "w") as f:
            json.dump(history, f, indent=2)
        lib.print_log(tag='SAVE', message=f"Saved history JSON to '{hist_json}'")

        # === Save summary ===
        summary["SEED"] = config['SEED']
        summary["model_path"] = model_path  # add model path reference
        summary["history_path"] = hist_json
        summary_path = os.path.join(save_dir, timestamp+"_summary.json")

        # ensure everything in summary is serializable
        safe_summary = {}
        for k, v in summary.items():
            if isinstance(v, torch.Tensor):
                safe_summary[k] = v.detach().cpu().item()
            elif isinstance(v, (list, tuple)):
                safe_summary[k] = [float(x.detach().cpu().item()) if isinstance(x, torch.Tensor) else x for x in v]
            else:
                safe_summary[k] = v

        with open(summary_path, "w") as f:
            json.dump(safe_summary, f, indent=2)
        lib.print_log(tag='SAVE', message=f"Saved summary to '{summary_path}'")


    duration = time.time() - run_start_time
    print("=" * width)
    print(">> ASSENT Run Finished <<".center(width))
    print(f"{'Execution time:':<20} {duration // 60:.0f} min and {duration % 60:.2f} seconds")
    print("=" * width)
