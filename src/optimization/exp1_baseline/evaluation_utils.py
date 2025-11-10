
import numpy as np
import matplotlib.pyplot as plt
import src.utils.optimization_utils as opt
from dataclasses import is_dataclass, asdict


def eval_joint_objective(G_comm_i, S_comm_i, G_sens_i, sol, alpha, lambda_cu=1.0, lambda_tg=1.0, interf_penalty=0.01):
    params = {'G_comm': G_comm_i, 'S_comm': S_comm_i, 'G_sens': G_sens_i, 'alpha': alpha,
              'lambda_cu': lambda_cu, 'lambda_tg': lambda_tg, 'interf_penalty':interf_penalty}
    milp_obj = opt.compute_milp_objective(params, sol)
    return milp_obj['obj_val']

def ecdf(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    v.sort()
    y = np.arange(1, len(v)+1) / len(v)
    return v, y


def to_sol_dict(sol):
    if is_dataclass(sol): return asdict(sol)
    if isinstance(sol, dict): return sol
    keys = ("tau","x","s","y_tx","y_rx")
    return {k: getattr(sol, k) for k in keys}


def group_by_alpha(alpha_vec, *flat_lists, N_per=None):
    """
    Split flat lists concatenated by alpha into per-alpha chunks.
    Returns:
        grouped: dict where grouped[a] = [G_comm_chunk, S_comm_chunk, G_sens_chunk, sol_chunk, ...]
        N_per:   number of realizations per alpha
    """
    n_alpha = len(alpha_vec)
    L = len(flat_lists[0])
    if N_per is None:
        if L % n_alpha != 0:
            raise ValueError(f"Total length {L} not divisible by #alphas {n_alpha}. "
                             f"Pass N_per explicitly.")
        N_per = L // n_alpha
    # sanity: all lists same length
    for idx, arr in enumerate(flat_lists):
        if len(arr) != L:
            raise ValueError(f"Input #{idx} length {len(arr)} != {L}")

    grouped = {}
    for i, a in enumerate(alpha_vec):
        s, e = i * N_per, (i + 1) * N_per
        grouped[a] = [arr[s:e] for arr in flat_lists]
    return grouped, N_per

def compute_joint_stats(alpha_vec, opt_objVal_list, N_per):
    """Median & IQR of the optimized joint objective at each alpha (no re-eval)."""
    comm_med, comm_q25, comm_q75 = [], [], []
    out_med, out_q25, out_q75 = [], [], []
    for i, a in enumerate(alpha_vec):
        s, e = i*N_per, (i+1)*N_per
        vals = np.asarray(opt_objVal_list[s:e], float)
        out_med.append(np.median(vals))
        out_q25.append(np.percentile(vals, 25))
        out_q75.append(np.percentile(vals, 75))
    return np.array(out_med), np.array(out_q25), np.array(out_q75)




def evaluate_with_alpha(G_comm_i, S_comm_i, G_sens_i, sol_dict, alpha,
                        lambda_cu=1.0, lambda_tg=1.0, interf_penalty=0.01) -> float:
    params = {'G_comm': G_comm_i, 'S_comm': S_comm_i, 'G_sens': G_sens_i, 'alpha': alpha,
              'lambda_cu': lambda_cu, 'lambda_tg': lambda_tg, 'interf_penalty': interf_penalty}
    milp_obj = opt.compute_milp_objective(params, sol_dict)
    return milp_obj['obj_val_noreward']


def compute_components_stats(alpha_vec, grouped, lambda_cu, lambda_tg, interf_penalty=0.01):
    """Re-evaluate each MILP solution at α=1 and α=0; return medians & IQR."""
    comm_med, comm_q25, comm_q75 = [], [], []
    sens_med, sens_q25, sens_q75 = [], [], []
    for a in alpha_vec:
        Gc, Sc, Gs, SOL = grouped[a]
        N = len(SOL)
        c = np.empty(N); c[:] = np.nan
        s = np.empty(N); s[:] = np.nan
        for i in range(N):
            sd = to_sol_dict(SOL[i])
            c[i] = evaluate_with_alpha(Gc[i], Sc[i], Gs[i], sd, alpha=1.0,
                                       lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=interf_penalty)
            s[i] = evaluate_with_alpha(Gc[i], Sc[i], Gs[i], sd, alpha=0.0,
                                       lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=interf_penalty)
        c = c[~np.isnan(c)]; s = s[~np.isnan(s)]
        comm_med.append(np.median(c)); comm_q25.append(np.percentile(c,25)); comm_q75.append(np.percentile(c,75))
        sens_med.append(np.median(s)); sens_q25.append(np.percentile(s,25)); sens_q75.append(np.percentile(s,75))
    return {
        "comm_med": np.array(comm_med), "comm_q25": np.array(comm_q25), "comm_q75": np.array(comm_q75),
        "sens_med": np.array(sens_med), "sens_q25": np.array(sens_q25), "sens_q75": np.array(sens_q75),
    }

def compute_structure_stats(alpha_vec, grouped):
    """Behavioral metrics vs α: tau TX fraction, RF share, UE coverage, target scheduling."""
    tx_frac, rf_comm_share, ue_cov_frac, tg_frac = [], [], [], []
    for a in alpha_vec:
        Gc, Sc, Gs, SOL = grouped[a]
        N = len(SOL)
        txf = []; rfs = []; ucf = []; tgf = []
        for i in range(N):
            s = to_sol_dict(SOL[i])
            N_ap = len(s["tau"])
            N_ue = s["x"].shape[1]
            N_tg = s["s"].size

            # TX fraction
            txf.append(s["tau"].sum() / max(N_ap,1))

            # RF share (comm vs sens) from usage counts
            comm_used = s["x"].sum()         # total comm links
            sens_used = s["y_tx"].sum() + s["y_rx"].sum()
            tot_used  = comm_used + sens_used
            rfs.append(comm_used / tot_used if tot_used > 0 else np.nan)

            # UE coverage fraction
            if N_ue > 0:
                ulinks_per_ue = (s["x"].sum(axis=0) > 0).astype(int)
                ucf.append(ulinks_per_ue.mean())
            else:
                ucf.append(np.nan)

            # Target scheduling fraction
            if N_tg > 0:
                tgf.append(s["s"].sum() / N_tg)
            else:
                tgf.append(np.nan)

        tx_frac.append(np.nanmean(txf))
        rf_comm_share.append(np.nanmean(rfs))
        ue_cov_frac.append(np.nanmean(ucf))
        tg_frac.append(np.nanmean(tgf))
    return {
        "tx_frac": np.array(tx_frac),
        "rf_comm_share": np.array(rf_comm_share),
        "ue_cov_frac": np.array(ue_cov_frac),
        "tg_frac": np.array(tg_frac),
    }

# ----------------------------
# PLOTS
# ----------------------------
def set_pub_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def plot_alpha_suite(alpha_vec, joint_stats, comp_stats, struct_stats):
    set_pub_rc()
    a = np.array(alpha_vec, float)
    J_med, J_q25, J_q75 = joint_stats
    # normalize components by endpoint medians (so both in [0,1])
    c_med = comp_stats["comm_med"].astype(float)
    s_med = comp_stats["sens_med"].astype(float)
    c_q25, c_q75 = comp_stats["comm_q25"], comp_stats["comm_q75"]
    s_q25, s_q75 = comp_stats["sens_q25"], comp_stats["sens_q75"]

    c_scale = max(c_med[-1], 1e-12)  # α=1
    s_scale = max(s_med[0],  1e-12)  # α=0
    cN_med, cN_q25, cN_q75 = c_med/c_scale, c_q25/c_scale, c_q75/c_scale
    sN_med, sN_q25, sN_q75 = s_med/s_scale, s_q25/s_scale, s_q75/s_scale

    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.4), constrained_layout=True)

    # (a) Joint objective vs α
    ax = axs[0,0]
    ax.plot(a, J_med, "-", color="tab:blue", lw=2.0, label="Joint objective (median)")
    ax.fill_between(a, J_q25, J_q75, color="tab:blue", alpha=0.2, lw=0)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("Optimized joint objective")
    ax.grid(True, ls=":", lw=0.8, alpha=0.35)
    ax.legend(frameon=False, loc="best")

    # (b) Normalized components vs α
    ax = axs[0,1]
    ax.plot(a, cN_med, "-",  color="tab:green", lw=2.0, label="Comm (median, norm)")
    ax.fill_between(a, cN_q25/c_scale, cN_q75/c_scale, color="tab:green", alpha=0.2, lw=0)
    ax.plot(a, sN_med, "-",  color="tab:orange", lw=2.0, label="Sens (median, norm)")
    ax.fill_between(a, sN_q25/s_scale, sN_q75/s_scale, color="tab:orange", alpha=0.2, lw=0)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("Utility (normalized to endpoint)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls=":", lw=0.8, alpha=0.35); ax.legend(frameon=False, loc="best")

    # (c) Resource split & roles vs α (two y-axes)
    ax = axs[1,0]
    ax2 = ax.twinx()
    ax.plot(a, struct_stats["tx_frac"], "-o", ms=3.5, color="tab:red", lw=1.8, label="TX AP fraction")
    ax.set_ylabel("TX AP fraction", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2.plot(a, struct_stats["rf_comm_share"], "-s", ms=3.5, color="tab:purple", lw=1.8, label="RF share (comm)")
    ax2.set_ylabel("RF share (comm)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_xlabel(r"$\alpha$")
    ax.grid(True, ls=":", lw=0.8, alpha=0.35)
    # compact legend
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines+lines2, labs+labs2, frameon=False, loc="best")

    # (d) Scheduling & coverage vs α
    ax = axs[1,1]
    ax.plot(a, struct_stats["tg_frac"], "-o", ms=3.5, color="tab:brown", lw=1.8, label="Targets scheduled (fraction)")
    ax.plot(a, struct_stats["ue_cov_frac"], "-^", ms=3.5, color="tab:cyan", lw=1.8, label="UE coverage (fraction)")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(True, ls=":", lw=0.8, alpha=0.35); ax.legend(frameon=False, loc="best")

    # fig.savefig("milp_alpha_suite.pdf", bbox_inches="tight")
    # fig.savefig("milp_alpha_suite.png", dpi=300, bbox_inches="tight")
    plt.show()





def _nanpercentiles(a, lo=25, hi=75):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(a)), float(np.percentile(a, lo)), float(np.percentile(a, hi))

def compute_structure_stats_mean_pctl(alpha_vec, grouped, p_lo=25, p_hi=75):
    """
    For each α: compute mean and [p_lo, p_hi] percentiles across realizations for:
      - UE coverage fraction
      - Targets scheduled fraction
      - TX AP fraction
    Returns arrays aligned with alpha_vec.
    """
    out = {k: [] for k in [
        "ue_mean","ue_lo","ue_hi",
        "tg_mean","tg_lo","tg_hi",
        "tx_mean","tx_lo","tx_hi",
        "rf_mean", "rf_lo", "rf_hi",
    ]}
    for a in alpha_vec:
        _, _, _, SOL = grouped[a]
        ue_vals, tg_vals, tx_vals, rf_vals = [], [], [], []
        for sol in SOL:
            s = to_sol_dict(sol)
            N_ap = len(s["tau"])
            N_ue = s["x"].shape[1]
            N_tg = s["s"].size

            tx_vals.append(s["tau"].sum() / max(N_ap, 1))

            if N_ue > 0:
                ue_cov = (s["x"].sum(axis=0) > 0).mean()
                ue_vals.append(ue_cov)
            if N_tg > 0:
                tg_frac = s["s"].sum() / N_tg
                tg_vals.append(tg_frac)
            # RF share (comm)
            comm_used = s["x"].sum()
            sens_used = s["y_tx"].sum() + s["y_rx"].sum()
            tot_used = comm_used + sens_used
            rf_vals.append(comm_used / tot_used if tot_used > 0 else np.nan)

        ue_m, ue_l, ue_h = _nanpercentiles(ue_vals, p_lo, p_hi)
        tg_m, tg_l, tg_h = _nanpercentiles(tg_vals, p_lo, p_hi)
        tx_m, tx_l, tx_h = _nanpercentiles(tx_vals, p_lo, p_hi)
        rf_m, rf_l, rf_h = _nanpercentiles(rf_vals, p_lo, p_hi)

        out["ue_mean"].append(ue_m); out["ue_lo"].append(ue_l); out["ue_hi"].append(ue_h)
        out["tg_mean"].append(tg_m); out["tg_lo"].append(tg_l); out["tg_hi"].append(tg_h)
        out["tx_mean"].append(tx_m); out["tx_lo"].append(tx_l); out["tx_hi"].append(tx_h)
        out["rf_mean"].append(rf_m); out["rf_lo"].append(rf_l); out["rf_hi"].append(rf_h)

    return {k: np.array(v, float) for k, v in out.items()}



