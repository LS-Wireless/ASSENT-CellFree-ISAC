
import numpy as np
import os
import json
import src.utils.library as lib
import src.utils.optimization_utils as opt
import matplotlib.pyplot as plt

from src.optimization.exp1_baseline.greedy_baseline import *

# Loading dataset

CONSOLE_RUN = True
cwd = os.getcwd()
console_run = cwd + '/src/optimization/exp1_baseline' if CONSOLE_RUN else ''

run_id = 'run_01'
folder_path = os.path.join(console_run, run_id)
metadata_path = os.path.join(folder_path, 'metadata.json')
with open(metadata_path) as f:
    metadata = json.load(f)
num_parts = metadata['config']['num_parts_to_save']

results = []
for i in range(num_parts):
    filename = f'2025-10-03_results_p{i+1}_of_{num_parts}.pkl'
    file_path = os.path.join(folder_path, filename)
    results += lib.load_results(file_path)

nsamps = len(results)
lib.print_log(tag='LOAD', message=f"Loaded results from '{folder_path}' with {nsamps} entries")


G_comm_list = [entry["G_comm"] for entry in results]
S_comm_list = [entry["S_comm"] for entry in results]
G_sens_list = [entry["G_sens"] for entry in results]
opt_objVal_list = [entry["opt_objVal"] for entry in results]
opt_solution_list = [entry["solution"] for entry in results]

#%% Loading coordinates (if needed)

coordinates_path = os.path.join(folder_path, 'coordinates.pkl')
coordinates = lib.load_results(coordinates_path)
user_pos_list = [entry["user_positions"] for entry in coordinates]

#%% Generating greedy solutions

alpha = metadata['config']['alpha']
lambda_cu = metadata['config']['lambda_cu']
lambda_tg = metadata['config']['lambda_tg']

greedy_solution_list = []
greedy_channel_solution_list = []
greedy_comm_only_solution_list = []
greedy_sens_only_solution_list = []
for i in range(nsamps):
    if (i+1) % 1000 == 0:
        lib.print_log(tag='RUN', message=f"Evaluating {i+1} of {nsamps}")
    G_comm = G_comm_list[i]
    G_sens = G_sens_list[i]
    max_gain = max(np.max(G_comm), np.max(G_sens))
    G_comm_norm = G_comm / max_gain
    G_sens_norm = G_sens / max_gain
    sol = greedy_assign_aligned_with_milp(G_comm=G_comm_norm, G_sens=G_sens_norm, N_rf=metadata['NetworkParams']['N_RF'],
                                          K_tx=metadata['config']['K_tx'], K_rx=metadata['config']['K_rx'],
                                          tau_milp=opt_solution_list[i].tau, s_milp=opt_solution_list[i].s,
                                          max_aps_per_user=10)
    greedy_solution_list.append(sol)
    sol_grd_channel = greedy_assign_channel_only(G_comm=G_comm_norm, G_sens=G_sens_norm, N_rf=metadata['NetworkParams']['N_RF'],
                                                 K_tx=metadata['config']['K_tx'], K_rx=metadata['config']['K_rx'],
                                                 max_aps_per_user=10, T_sched=None)
    greedy_channel_solution_list.append(sol_grd_channel)
    sol_comm_only = greedy_assign_comm_only(G_comm=G_comm_norm, N_rf=metadata['NetworkParams']['N_RF'], N_tg=metadata['NetworkParams']['N_tg'], max_aps_per_user=10)
    greedy_comm_only_solution_list.append(sol_comm_only)
    sol_sens_only = greedy_assign_sens_only(G_sens=G_sens_norm, N_rf=metadata['NetworkParams']['N_RF'],
                                            K_tx=metadata['config']['K_tx'], K_rx=metadata['config']['K_rx'],
                                            N_ue=G_comm.shape[1], T_sched=None)
    greedy_sens_only_solution_list.append(sol_sens_only)

lib.print_log(tag='RUN', message=f"Finished generating {nsamps} greedy solutions!")


def ecdf(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    v.sort()
    y = np.arange(1, len(v)+1) / len(v)
    return v, y

def eval_joint_objective(G_comm_i, S_comm_i, G_sens_i, sol):
    params = {'G_comm': G_comm_i, 'S_comm': S_comm_i, 'G_sens': G_sens_i, 'alpha': alpha,
              'lambda_cu': lambda_cu, 'lambda_tg': lambda_tg, 'interf_penalty': metadata['config']['interf_penalty']}
    milp_obj = opt.compute_milp_objective(params, sol)
    return milp_obj['obj_val']

# --- Compute greedy objective for each realization ---
greedy_obj = np.empty(nsamps, dtype=float)
greedy_ch_obj = np.empty(nsamps, dtype=float)
greedy_comm_obj = np.empty(nsamps, dtype=float)
greedy_sens_obj = np.empty(nsamps, dtype=float)

for i in range(nsamps):
    greedy_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_solution_list[i])
    greedy_ch_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_channel_solution_list[i])
    greedy_comm_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_comm_only_solution_list[i])
    greedy_sens_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_sens_only_solution_list[i])

lib.print_log(tag='RUN', message=f"Finished computing greedy objective for {nsamps} realizations!")

# --- Build ECDFs ---
x_milp, y_milp = ecdf(10*np.log10(opt_objVal_list))
x_grdy, y_grdy = ecdf(10*np.log10(greedy_obj))
x_ch, y_ch = ecdf(10*np.log10(greedy_ch_obj))
x_comm, y_comm = ecdf(10*np.log10(greedy_comm_obj))
x_comm_offset = -0.1
x_sens, y_sens = ecdf(10*np.log10(greedy_sens_obj))

lib.print_log(tag='RUN', message=f"Finished building ECDFs!")


#%% Plotting

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
def x_at_percentile(x, y, p=0.5):
    p = float(np.clip(p, 0.0, 1.0))
    return float(np.interp(p, y, x))

lw = 1.8
font_size = 12
colors = ['tab:green', 'tab:blue', 'tab:red']
palette = {
    "milp":      "tab:blue",
    "greedy":    "tab:red",
    "ch_only":   "tab:orange",
    "comm_only": "tab:green",
    "sens_only": "tab:purple",
}

dashes = {
    "milp":         '-',
    "greedy":       '--',
    "ch_only":      '-.',
    "comm_only":    '-.',
    "sens_only":    '-.',
}

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,          # base font size
    'axes.labelsize': font_size,     # x and y labels
    'xtick.labelsize': font_size,    # x ticks
    'ytick.labelsize': font_size,    # y ticks
    'legend.fontsize': 11,    # legend
    'axes.titlesize': 10,    # title (if you have one)
})

fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(x_milp, y_milp,    linestyle=dashes['milp'],        color=palette['milp'],      lw=lw, label=r'MILP ($\alpha$=0.6)')
ax.plot(x_grdy, y_grdy,    linestyle=dashes['greedy'],      color=palette['greedy'],    lw=lw, label=r'Greedy ($\alpha$=0.6)')
ax.plot(x_ch,   y_ch,      linestyle=dashes['ch_only'],     color=palette['ch_only'],   lw=lw, label='Greedy (channel-only)')
ax.plot(x_comm+x_comm_offset, y_comm,linestyle=dashes['comm_only'],   color=palette['comm_only'], lw=lw, label='Greedy (comm-only)')
ax.plot(x_sens, y_sens,    linestyle=dashes['sens_only'],   color=palette['sens_only'], lw=lw, label='Greedy (sens-only)')

axins = zoomed_inset_axes(ax, zoom=2.2, loc="upper left")  # try "lower right" if overlap
axins.set_facecolor("white")

# re-plot in the inset
axins.plot(x_milp, y_milp, linestyle=dashes["milp"],      color=palette["milp"],      lw=lw)
axins.plot(x_ch,   y_ch,   linestyle=dashes["ch_only"],   color=palette["ch_only"],   lw=lw)
axins.plot(x_comm+x_comm_offset, y_comm, linestyle=dashes["comm_only"], color=palette["comm_only"], lw=lw)
axins.plot(x_sens, y_sens, linestyle=dashes["sens_only"], color=palette["sens_only"], lw=lw)
axins.plot(x_grdy, y_grdy, linestyle=dashes["greedy"],    color=palette["greedy"],    lw=lw)

# focus window (e.g., between medians of the 4 greedy curves)
x_left  = 2.5
x_right = 3.5

# set EXACT margins
x_margin_left  = 0.10    # e.g., 0.10 objective units
x_margin_right = 0.10
y_bottom = 0.55          # lower CDF bound
y_top    = 0.80          # upper CDF bound

axins.set_xlim(x_left - x_margin_left, x_right + x_margin_right)
axins.set_ylim(y_bottom, y_top)


axins.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.4)
# hide inset tick labels
for t in axins.get_xticklabels() + axins.get_yticklabels():
    t.set_visible(False)
axins.tick_params(axis="both", which="both", length=0)

# draw rectangle + connectors
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=1.0, alpha=0.9)


# Percentile markers (optional)
p5_milp  = np.percentile(x_milp, 5)
p50_milp = np.percentile(x_milp, 50)
p95_milp  = np.percentile(x_milp, 95)

if Vlines := False:
    plt.axvline(p5_milp,  linestyle=':', linewidth=1.5, color=colors[1])
    plt.axvline(p50_milp, linestyle=':', linewidth=1.5, color=colors[1])
    plt.axvline(p95_milp,  linestyle=':', linewidth=1.5, color=colors[1])

ax.set_xlim(-1, 7)
ax.set_ylim(0, 1)
ax.set_xlabel('Joint Objective Value (dB)')
ax.set_ylabel('Empirical CDF')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.tight_layout(pad=0.8)
plt.show()
plt.rcdefaults()

# Save a camera-ready PDF/PNG
# plt.savefig('ecdf_milp_vs_greedy_alpha0p6.pdf')
# plt.savefig('ecdf_milp_vs_greedy_alpha0p6.png', dpi=300)

# --- Print quick stats for caption ---
def stats(name, arr):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    return (f"{name}: mean={np.mean(arr):.3f}, "
            f"p5={np.percentile(arr,5):.3f}, "
            f"median={np.percentile(arr,50):.3f}, "
            f"p95={np.percentile(arr,95):.3f}, N={len(arr)}")

print(stats("MILP",  opt_objVal_list))
print(stats("Greedy", greedy_obj))
impr_p50 = (np.percentile(x_milp,50) - np.percentile(x_grdy,50))
impr_p5  = (np.percentile(x_milp,5)  - np.percentile(x_grdy,5))
print(f"Δ median = {impr_p50:.3f}, Δ 5th-pct = {impr_p5:.3f}")



#%% Loading dataset from run_02

CONSOLE_RUN = True
cwd = os.getcwd()
console_run = cwd + '/src/optimization/exp1_baseline' if CONSOLE_RUN else ''

run_id = 'run_02'
folder_path = os.path.join(console_run, run_id)
metadata_path = os.path.join(folder_path, 'metadata.json')
with open(metadata_path) as f:
    metadata_02 = json.load(f)
num_parts = metadata_02['config']['num_parts_to_save']
pos_reals = metadata_02['config']['num_entity_position_realizations']
ch_reals = metadata_02['config']['num_entity_channel_realizations']
alpha_vec = metadata_02['config']['alpha']
nsamps_per_alpha = pos_reals * ch_reals

results_02 = []
for i in range(num_parts):
    filename = f'2025-10-05_results_p{i+1}_of_{num_parts}.pkl'
    file_path = os.path.join(folder_path, filename)
    results_02 += lib.load_results(file_path)

nsamps_02 = len(results_02)
lib.print_log(tag='LOAD', message=f"Loaded results from '{folder_path}' with {nsamps_02} entries")

G_comm_list_02 = [entry["G_comm"] for entry in results_02]
S_comm_list_02 = [entry["S_comm"] for entry in results_02]
G_sens_list_02 = [entry["G_sens"] for entry in results_02]
opt_objVal_list_02 = [entry["opt_objVal"] for entry in results_02]
opt_solution_list_02 = [entry["solution"] for entry in results_02]


#%%


def evaluate_with_alpha(G_comm_i, S_comm_i, G_sens_i, sol_dict, alpha: float) -> float:
    params = {'G_comm': G_comm_i, 'S_comm': S_comm_i, 'G_sens': G_sens_i, 'alpha': alpha,
              'lambda_cu': metadata_02['config']['lambda_cu'], 'lambda_tg': metadata_02['config']['lambda_tg'],
              'interf_penalty': metadata_02['config']['interf_penalty']}
    milp_obj = opt.compute_milp_objective(params, sol_dict)
    return milp_obj['obj_val_noreward']

# ----------------------------
# helpers
# ----------------------------
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

def compute_components_stats(alpha_vec, grouped):
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
            c[i] = evaluate_with_alpha(Gc[i], Sc[i], Gs[i], sd, alpha=1.0)
            s[i] = evaluate_with_alpha(Gc[i], Sc[i], Gs[i], sd, alpha=0.0)
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

# ----------------------------
# USE WITH THE DATA
# ----------------------------
# alpha_vec: 11 values from 0.0 to 1.0
# Each *_list_02 has length 22000 concatenated by α in 2000-sized blocks
# G_comm_list_02, S_comm_list_02, G_sens_list_02, opt_solution_list_02, opt_objVal_list_02

grouped, N_per = group_by_alpha(
    alpha_vec,
    G_comm_list_02, S_comm_list_02, G_sens_list_02, opt_solution_list_02
)

# Then keep using the rest of the pipeline you already have:
J_med, J_q25, J_q75 = compute_joint_stats(alpha_vec, opt_objVal_list_02, N_per)
comp_stats  = compute_components_stats(alpha_vec, grouped)
struct_stats = compute_structure_stats(alpha_vec, grouped)
plot_alpha_suite(alpha_vec, (J_med, J_q25, J_q75), comp_stats, struct_stats)


#%%

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import is_dataclass, asdict

# ---------- helpers ----------
def to_sol_dict(sol):
    if is_dataclass(sol): return asdict(sol)
    if isinstance(sol, dict): return sol
    keys = ("tau","x","s","y_tx","y_rx")
    return {k: getattr(sol, k) for k in keys}

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



# We already have `grouped` from group_by_alpha(...)
p_lo, p_hi = 25, 75
struct_stats_p = compute_structure_stats_mean_pctl(alpha_vec, grouped, p_lo=p_lo, p_hi=p_hi)
p_label = f"{p_lo}-{p_hi}th-pct"

a = np.asarray(alpha_vec, float)
order = np.argsort(a)
a = a[order]
# sort all series to match a
def srt(k): return np.asarray(struct_stats_p[k], float)[order]
ue_mean, ue_lo, ue_hi = srt("ue_mean"), srt("ue_lo"), srt("ue_hi")
tg_mean, tg_lo, tg_hi = srt("tg_mean"), srt("tg_lo"), srt("tg_hi")
tx_mean, tx_lo, tx_hi = srt("tx_mean"), srt("tx_lo"), srt("tx_hi")
rf_mean, rf_lo, rf_hi = srt("rf_mean"), srt("rf_lo"), srt("rf_hi")

# default colors (Tableau tabs)
colors = {"ue": "tab:blue", "tg": "tab:green", "tx": "tab:red", "rf": "tab:purple"}

# style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, axL = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
axR = axL.twinx()
# ensure right axis doesn't cover left-axis shading
axR.set_zorder(axL.get_zorder() + 1)
axR.patch.set_visible(False)

# masks for shading (avoid NaNs)
m_ue = np.isfinite(ue_lo) & np.isfinite(ue_hi)
m_tg = np.isfinite(tg_lo) & np.isfinite(tg_hi)
m_tx = np.isfinite(tx_lo) & np.isfinite(tx_hi)
m_rf = np.isfinite(rf_lo) & np.isfinite(rf_hi)

# --- Left axis: percentile shading + mean lines ---
if m_ue.any():
    axL.fill_between(a[m_ue], ue_lo[m_ue], ue_hi[m_ue],
                     color=colors["ue"], alpha=0.20, lw=0, zorder=1.0)
if m_tg.any():
    axL.fill_between(a[m_tg], tg_lo[m_tg], tg_hi[m_tg],
                     color=colors["tg"], alpha=0.20, lw=0, zorder=1.0)

ln1, = axL.plot(a, ue_mean, "-o", ms=4, lw=2.0, color=colors["ue"], label="UE coverage (mean)", zorder=2.0)
ln2, = axL.plot(a, tg_mean, "-^", ms=4, lw=2.0, color=colors["tg"], label="Targets scheduled (mean)", zorder=2.0)

axL.set_xlabel(r"$\alpha$ Values")
axL.set_ylabel("Coverage / Targets (fraction)")
axL.set_ylim(0.0, 1.05)
axL.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.35)

# --- Right axis: percentile shading + mean line ---
if m_tx.any():
    axR.fill_between(a[m_tx], tx_lo[m_tx], tx_hi[m_tx],
                     color=colors["tx"], alpha=0.18, lw=0, zorder=1.1)
if m_rf.any():
    axR.fill_between(a[m_rf], rf_lo[m_rf], rf_hi[m_rf], color=colors["rf"], alpha=0.18, lw=0, zorder=1.1,
                     label=f"RF share (comm) ({p_label})")

ln3, = axR.plot(a, tx_mean, "-s", ms=4, lw=2.0, color=colors["tx"], label="TX AP fraction (mean)", zorder=2.1)
ln4, = axR.plot(a, rf_mean, "-D", ms=4, lw=2.0, color=colors["rf"], label="RF share (comm) (mean)", zorder=2.1)

axR.set_ylabel("TX AP / RF share fraction", color=colors["tx"])
axR.tick_params(axis="y", labelcolor=colors["tx"])
axR.set_ylim(0.0, 1.05)

# combined legend
lines = [ln1, ln2, ln3, ln4]
axL.legend(lines, [l.get_label() for l in lines], frameon=False, loc="best", handlelength=2.6)

# plt.tight_layout()
plt.show()



#%% Final figure generation script

SAVE_FIGS = True
CONSOLE_RUN = True
cwd = os.getcwd()
console_run = cwd + '/src/optimization/exp1_baseline' if CONSOLE_RUN else ''
save_path = os.path.join(console_run, 'figures')
os.makedirs(save_path, exist_ok=True)

lw = 1.8
font_size = 12
colors = ['tab:green', 'tab:blue', 'tab:red']
palette = {
    "milp":      "tab:blue",
    "greedy":    "tab:red",
    "ch_only":   "tab:orange",
    "comm_only": "tab:green",
    "sens_only": "tab:purple",
}

dashes = {
    "milp":         '-',
    "greedy":       '--',
    "ch_only":      '-.',
    "comm_only":    '-.',
    "sens_only":    '-.',
}

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,          # base font size
    'axes.labelsize': font_size,     # x and y labels
    'xtick.labelsize': font_size,    # x ticks
    'ytick.labelsize': font_size,    # y ticks
    'legend.fontsize': 11,    # legend
    'axes.titlesize': 10,    # title
})

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)

ax1.plot(x_milp, y_milp,    linestyle=dashes['milp'],        color=palette['milp'],      lw=lw, label=r'MILP ($\alpha$=0.6)')
ax1.plot(x_grdy, y_grdy,    linestyle=dashes['greedy'],      color=palette['greedy'],    lw=lw, label=r'Greedy (MILP-aligned)')
ax1.plot(x_ch,   y_ch,      linestyle=dashes['ch_only'],     color=palette['ch_only'],   lw=lw, label='Greedy (channel-only)')
ax1.plot(x_comm+x_comm_offset, y_comm,linestyle=dashes['comm_only'],   color=palette['comm_only'], lw=lw, label='Greedy (comm-only)')
ax1.plot(x_sens, y_sens,    linestyle=dashes['sens_only'],   color=palette['sens_only'], lw=lw, label='Greedy (sens-only)')




axins = zoomed_inset_axes(ax1, zoom=2.2, loc="upper left")  # try "lower right" if overlap
axins.set_facecolor("white")

# re-plot in the inset
axins.plot(x_milp, y_milp, linestyle=dashes["milp"],      color=palette["milp"],      lw=lw)
axins.plot(x_ch,   y_ch,   linestyle=dashes["ch_only"],   color=palette["ch_only"],   lw=lw)
axins.plot(x_comm-0.1, y_comm, linestyle=dashes["comm_only"], color=palette["comm_only"], lw=lw)
axins.plot(x_sens, y_sens, linestyle=dashes["sens_only"], color=palette["sens_only"], lw=lw)
axins.plot(x_grdy, y_grdy, linestyle=dashes["greedy"],    color=palette["greedy"],    lw=lw)

# focus window (e.g., between medians of the 4 greedy curves)
x_left  = 2.5
x_right = 3.5

# set EXACT margins
x_margin_left  = 0.10    # e.g., 0.10 objective units
x_margin_right = 0.10
y_bottom = 0.55          # lower CDF bound
y_top    = 0.80          # upper CDF bound

axins.set_xlim(x_left - x_margin_left, x_right + x_margin_right)
axins.set_ylim(y_bottom, y_top)


axins.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.4)
# hide inset tick labels
for t in axins.get_xticklabels() + axins.get_yticklabels():
    t.set_visible(False)
axins.tick_params(axis="both", which="both", length=0)

# draw rectangle + connectors
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=1.0, alpha=0.9)


ax1.set_xlim(-1, 7)
ax1.set_ylim(0, 1)
ax1.set_xlabel('Joint Objective Value (dB)')
ax1.set_ylabel('Empirical CDF')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)




axL = fig.add_subplot(1, 2, 2)
axR = axL.twinx()
# ensure right axis doesn't cover left-axis shading
axR.set_zorder(axL.get_zorder() + 1)
axR.patch.set_visible(False)

# default colors (Tableau tabs)
colors = {"ue": "tab:blue", "tg": "tab:green", "tx": "tab:red", "rf": "tab:purple"}
msize = 5.0

# masks for shading (avoid NaNs)
m_ue = np.isfinite(ue_lo) & np.isfinite(ue_hi)
m_tg = np.isfinite(tg_lo) & np.isfinite(tg_hi)
m_tx = np.isfinite(tx_lo) & np.isfinite(tx_hi)
m_rf = np.isfinite(rf_lo) & np.isfinite(rf_hi)

# --- Left axis: percentile shading + mean lines ---
if m_ue.any():
    axL.fill_between(a[m_ue], ue_lo[m_ue], ue_hi[m_ue],
                     color=colors["ue"], alpha=0.20, lw=0, zorder=1.0)
if m_tg.any():
    axL.fill_between(a[m_tg], tg_lo[m_tg], tg_hi[m_tg],
                     color=colors["tg"], alpha=0.20, lw=0, zorder=1.0)

ln1, = axL.plot(a, ue_mean, "-o", ms=msize, lw=2.0, color=colors["ue"], label="UE coverage", zorder=2.0)
ln2, = axL.plot(a, tg_mean, "-^", ms=msize, lw=2.0, color=colors["tg"], label="Targets scheduled", zorder=2.0)

axL.set_xlabel(r"$\alpha$ Values")
axL.set_ylabel("Coverage / Targets (fraction)")
axL.set_ylim(0.0, 1.05)
axL.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.35)

# --- Right axis: percentile shading + mean line ---
if m_tx.any():
    axR.fill_between(a[m_tx], tx_lo[m_tx], tx_hi[m_tx],
                     color=colors["tx"], alpha=0.18, lw=0, zorder=1.1)
if m_rf.any():
    axR.fill_between(a[m_rf], rf_lo[m_rf], rf_hi[m_rf], color=colors["rf"], alpha=0.18, lw=0, zorder=1.1,
                     label=f"RF share (comm) ({p_label})")

ln3, = axR.plot(a, tx_mean, "-s", ms=msize, lw=2.0, color=colors["tx"], label="TX AP fraction", zorder=2.1)
ln4, = axR.plot(a, rf_mean, "-D", ms=msize, lw=2.0, color=colors["rf"], label="RF share (comm)", zorder=2.1)

axR.set_ylabel("TX AP / RF share fraction", color=colors["tx"])
axR.tick_params(axis="y", labelcolor=colors["tx"])
axR.set_ylim(0.0, 1.05)

# combined legend
lines = [ln1, ln2, ln3, ln4]
axL.legend(lines, [l.get_label() for l in lines], frameon=True, loc="best", handlelength=2.6, fontsize=12)

plt.tight_layout(pad=0.8)
plt.show()
plt.rcdefaults()

# Save a camera-ready PDF/PNG
if SAVE_FIGS:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = os.path.join(save_path, timestamp + f"_milp_vs_greedy_and_coverage_fraction_{p_label}.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
    filename = os.path.join(save_path, timestamp + f"_milp_vs_greedy_and_coverage_fraction_{p_label}.pdf")
    fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
    lib.print_log(tag='SAVE', message=f"Saved figures to '{save_path}'")


