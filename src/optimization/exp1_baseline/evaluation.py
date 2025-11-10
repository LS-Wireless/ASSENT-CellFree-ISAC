

import os
import json
import src.utils.library as lib

from src.optimization.exp1_baseline.greedy_baseline import *
from src.optimization.exp1_baseline.evaluation_utils import *

# Loading dataset

CONSOLE_RUN = True
cwd = os.getcwd()
console_run = cwd + '/src/optimization/exp1_baseline' if CONSOLE_RUN else ''

run_id = 'run_03'
folder_path = os.path.join(console_run, run_id)
metadata_path = os.path.join(folder_path, 'metadata.json')
with open(metadata_path) as f:
    metadata = json.load(f)
num_parts = metadata['config']['num_parts_to_save']
parts_to_load = [2, 3]

all_results = []
for i in parts_to_load:
    filename = f'2025-11-06_results_p{i}_of_{num_parts}.pkl'
    file_path = os.path.join(folder_path, filename)
    all_results += lib.load_results(file_path)
    lib.print_log(tag='LOAD', message=f"Loading '{file_path}'")

results = all_results[:10000]
results_2ndAlpha = all_results[10000:]
nsamps = len(results)
lib.print_log(tag='LOAD', message=f"Loaded results from '{folder_path}' with {nsamps} entries")


G_comm_list = [entry["G_comm"] for entry in results]
S_comm_list = [entry["S_comm"] for entry in results]
G_sens_list = [entry["G_sens"] for entry in results]

opt_objVal_list = [entry["opt_objVal"] for entry in results]
opt_solution_list = [entry["solution"] for entry in results]

opt_objVal_list_2ndAlpha = [entry["opt_objVal"] for entry in results_2ndAlpha]
opt_solution_list_2ndAlpha = [entry["solution"] for entry in results_2ndAlpha]

# Loading coordinates (if needed)
# coordinates_path = os.path.join(folder_path, 'coordinates.pkl')
# coordinates = lib.load_results(coordinates_path)
# user_pos_list = [entry["user_positions"] for entry in coordinates]

#%% Generating greedy solutions (first alpha value)

alpha = metadata['config']['alpha'][parts_to_load[0]-1]
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

# --- Compute greedy objective for each realization ---
greedy_obj = np.empty(nsamps, dtype=float)
greedy_ch_obj = np.empty(nsamps, dtype=float)
greedy_comm_obj = np.empty(nsamps, dtype=float)
greedy_sens_obj = np.empty(nsamps, dtype=float)

for i in range(nsamps):
    greedy_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_solution_list[i], alpha=alpha,
                                         lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=metadata['config']['interf_penalty'])
    greedy_ch_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_channel_solution_list[i], alpha=alpha,
                                            lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=metadata['config']['interf_penalty'])
    greedy_comm_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_comm_only_solution_list[i], alpha=alpha,
                                              lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=metadata['config']['interf_penalty'])
    greedy_sens_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_sens_only_solution_list[i], alpha=alpha,
                                              lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=metadata['config']['interf_penalty'])

lib.print_log(tag='RUN', message=f"Finished computing greedy objective for {nsamps} realizations!")

# --- Build ECDFs ---
x_milp, y_milp = ecdf(10*np.log10(opt_objVal_list))
x_grdy, y_grdy = ecdf(10*np.log10(greedy_obj))
x_ch, y_ch = ecdf(10*np.log10(greedy_ch_obj))
x_comm, y_comm = ecdf(10*np.log10(greedy_comm_obj))
x_comm_offset = -0.1
x_sens_offset = -0.2
x_sens, y_sens = ecdf(10*np.log10(greedy_sens_obj))

lib.print_log(tag='RUN', message=f"Finished building ECDFs!")


#%% Generating greedy solutions (second alpha value)

alpha_2ndAlpha = metadata['config']['alpha'][parts_to_load[1]-1]

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

# --- Compute greedy objective for each realization ---
greedy_obj = np.empty(nsamps, dtype=float)
greedy_ch_obj = np.empty(nsamps, dtype=float)
greedy_comm_obj = np.empty(nsamps, dtype=float)
greedy_sens_obj = np.empty(nsamps, dtype=float)

for i in range(nsamps):
    greedy_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_solution_list[i], alpha=alpha_2ndAlpha,
                                         lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=metadata['config']['interf_penalty'])
    greedy_ch_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_channel_solution_list[i], alpha=alpha_2ndAlpha,
                                            lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=metadata['config']['interf_penalty'])
    greedy_comm_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_comm_only_solution_list[i], alpha=alpha_2ndAlpha,
                                              lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=metadata['config']['interf_penalty'])
    greedy_sens_obj[i] = eval_joint_objective(G_comm_list[i], S_comm_list[i], G_sens_list[i], greedy_sens_only_solution_list[i], alpha=alpha_2ndAlpha,
                                              lambda_cu=lambda_cu, lambda_tg=lambda_tg, interf_penalty=metadata['config']['interf_penalty'])

lib.print_log(tag='RUN', message=f"Finished computing greedy objective for {nsamps} realizations!")

# --- Build ECDFs ---
x_milp_2ndAlpha, y_milp_2ndAlpha = ecdf(10*np.log10(opt_objVal_list_2ndAlpha))
x_grdy_2ndAlpha, y_grdy_2ndAlpha = ecdf(10*np.log10(greedy_obj))
x_ch_2ndAlpha, y_ch_2ndAlpha = ecdf(10*np.log10(greedy_ch_obj))
x_comm_2ndAlpha, y_comm_2ndAlpha = ecdf(10*np.log10(greedy_comm_obj))
x_sens_2ndAlpha, y_sens_2ndAlpha = ecdf(10*np.log10(greedy_sens_obj))

lib.print_log(tag='RUN', message=f"Finished building ECDFs for second alpha value!")


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
    "milp":         '--',
    "greedy":       '--',
    "ch_only":      '--',
    "comm_only":    '--',
    "sens_only":    '--',

    "milp2":         '-',
    "greedy2":       '-',
    "ch_only2":      '-',
    "comm_only2":    '-',
    "sens_only2":    '-',
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

ax.plot(x_milp, y_milp,    linestyle=dashes['milp'],        color=palette['milp'],      lw=lw, label=rf'MILP ($\alpha$={alpha})')
ax.plot(x_grdy, y_grdy,    linestyle=dashes['greedy'],      color=palette['greedy'],    lw=lw, label=rf'Grdy ($\alpha$={alpha})')
ax.plot(x_ch,   y_ch,      linestyle=dashes['ch_only'],     color=palette['ch_only'],   lw=lw, label='Grdy (channel-only)')
ax.plot(x_comm+x_comm_offset, y_comm,linestyle=dashes['comm_only'],   color=palette['comm_only'], lw=lw, label='Grdy (comm-only)')
ax.plot(x_sens, y_sens,    linestyle=dashes['sens_only'],   color=palette['sens_only'], lw=lw, label='Grdy (sens-only)')

ax.plot(x_milp_2ndAlpha, y_milp_2ndAlpha,    linestyle=dashes['milp2'],        color=palette['milp'],      lw=lw, label=rf'MILP ($\alpha$={alpha_2ndAlpha})')
ax.plot(x_grdy_2ndAlpha, y_grdy_2ndAlpha,    linestyle=dashes['greedy2'],      color=palette['greedy'],    lw=lw, label=rf'Grdy ($\alpha$={alpha_2ndAlpha})')
ax.plot(x_ch_2ndAlpha,   y_ch_2ndAlpha,      linestyle=dashes['ch_only2'],     color=palette['ch_only'],   lw=lw, label='Grdy (channel-only)')
ax.plot(x_comm_2ndAlpha+x_comm_offset, y_comm_2ndAlpha,linestyle=dashes['comm_only2'],   color=palette['comm_only'], lw=lw, label='Grdy (comm-only)')
ax.plot(x_sens_2ndAlpha+x_sens_offset, y_sens_2ndAlpha,    linestyle=dashes['sens_only2'],   color=palette['sens_only'], lw=lw, label='Grdy (sens-only)')

axins = zoomed_inset_axes(ax, zoom=4, loc="upper right")  # try "lower right" if overlap
axins.set_facecolor("white")

# re-plot in the inset
axins.plot(x_milp, y_milp, linestyle=dashes["milp"],      color=palette["milp"],      lw=lw)
axins.plot(x_ch,   y_ch,   linestyle=dashes["ch_only"],   color=palette["ch_only"],   lw=lw)
axins.plot(x_comm+x_comm_offset, y_comm, linestyle=dashes["comm_only"], color=palette["comm_only"], lw=lw)
axins.plot(x_sens, y_sens, linestyle=dashes["sens_only"], color=palette["sens_only"], lw=lw)
axins.plot(x_grdy, y_grdy, linestyle=dashes["greedy"],    color=palette["greedy"],    lw=lw)

# focus window (e.g., between medians of the 4 greedy curves)
x_left  = 1.7
x_right = 1.9

# set EXACT margins
x_margin_left  = 0.10    # e.g., 0.10 objective units
x_margin_right = 0.10
y_bottom = 0.6         # lower CDF bound
y_top    = 0.8          # upper CDF bound

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

ax.set_xlim(-3, 5)
ax.set_ylim(0, 1)
ax.set_xlabel('Joint Objective Value (dB)')
ax.set_ylabel('Empirical CDF')
ax.legend(loc='best')
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



#%% Loading dataset from run_02 (Previous Version)

pareto_version = 1
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


#%% Loading dataset from exp1_pareto (New Version)

pareto_version = 2
CONSOLE_RUN = True
cwd = os.getcwd()
console_run = cwd + '/src/optimization/exp1_pareto' if CONSOLE_RUN else ''

run_id = 'run_01'
folder_path = os.path.join(console_run, run_id)
metadata_path = os.path.join(folder_path, 'metadata.json')
with open(metadata_path) as f:
    metadata_02 = json.load(f)
num_parts = metadata_02['config']['num_parts_to_save']
alpha_vec = metadata_02['config']['alpha']
lib.print_log(tag='LOAD', message=f"Loaded alpha values: '{alpha_vec}'")

if num_parts == 1:
    filename = f'2025-11-07-12-03_results.pkl'
    file_path = os.path.join(folder_path, filename)
    results_02 = lib.load_results(file_path)
else:
    results_02 = []
    for i in range(num_parts):
        filename = f'2025-11-07_results_p{i+1}_of_{num_parts}.pkl'
        file_path = os.path.join(folder_path, filename)
        results_02 += lib.load_results(file_path)

nsamps_02 = len(results_02)
lib.print_log(tag='LOAD', message=f"Loaded results from '{folder_path}' with {nsamps_02} entries")

G_comm_list_02 = G_comm_list * len(alpha_vec)
S_comm_list_02 = S_comm_list * len(alpha_vec)
G_sens_list_02 = G_sens_list * len(alpha_vec)
opt_objVal_list_02 = [entry["opt_objVal"] for entry in results_02]
opt_solution_list_02 = [entry["solution"] for entry in results_02]

grouped, N_per = group_by_alpha(alpha_vec,
                                G_comm_list_02, S_comm_list_02, G_sens_list_02, opt_solution_list_02)

#%% Plot Alpha Suite

# ----------------------------
# USE WITH THE DATA
# ----------------------------
# alpha_vec: 11 values from 0.0 to 1.0
# G_comm_list_02, S_comm_list_02, G_sens_list_02, opt_solution_list_02, opt_objVal_list_02


compute_components_stats(alpha_vec, grouped, lambda_cu, lambda_tg, interf_penalty=0.01)

# Then keep using the rest of the pipeline you already have:
J_med, J_q25, J_q75 = compute_joint_stats(alpha_vec, opt_objVal_list_02, N_per)
comp_stats  = compute_components_stats(alpha_vec, grouped, lambda_cu=metadata_02['config']['lambda_cu'],
                                       lambda_tg=metadata_02['config']['lambda_tg'], interf_penalty=metadata_02['config']['interf_penalty'])
struct_stats = compute_structure_stats(alpha_vec, grouped)
plot_alpha_suite(alpha_vec, (J_med, J_q25, J_q75), comp_stats, struct_stats)


#%%


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



#%% Final figure generation script (Version 1)

SAVE_FIGS = False
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

ax1.plot(x_milp, y_milp,    linestyle=dashes['milp'],        color=palette['milp'],      lw=lw, label=rf'MILP ($\alpha$={alpha})')
ax1.plot(x_grdy, y_grdy,    linestyle=dashes['greedy'],      color=palette['greedy'],    lw=lw, label=r'Greedy (MILP-aligned)')
ax1.plot(x_ch,   y_ch,      linestyle=dashes['ch_only'],     color=palette['ch_only'],   lw=lw, label='Greedy (channel-only)')
ax1.plot(x_comm+x_comm_offset, y_comm,linestyle=dashes['comm_only'],   color=palette['comm_only'], lw=lw, label='Greedy (comm-only)')
ax1.plot(x_sens, y_sens,    linestyle=dashes['sens_only'],   color=palette['sens_only'], lw=lw, label='Greedy (sens-only)')




axins = zoomed_inset_axes(ax1, zoom=4, loc="upper right")  # try "lower right" if overlap
axins.set_facecolor("white")

# re-plot in the inset
axins.plot(x_milp, y_milp, linestyle=dashes["milp"],      color=palette["milp"],      lw=lw)
axins.plot(x_ch,   y_ch,   linestyle=dashes["ch_only"],   color=palette["ch_only"],   lw=lw)
axins.plot(x_comm+x_comm_offset, y_comm, linestyle=dashes["comm_only"], color=palette["comm_only"], lw=lw)
axins.plot(x_sens, y_sens, linestyle=dashes["sens_only"], color=palette["sens_only"], lw=lw)
axins.plot(x_grdy, y_grdy, linestyle=dashes["greedy"],    color=palette["greedy"],    lw=lw)

# focus window (e.g., between medians of the 4 greedy curves)
x_left  = 1.7
x_right = 1.9

# set EXACT margins
x_margin_left  = 0.10    # e.g., 0.10 objective units
x_margin_right = 0.10
y_bottom = 0.6          # lower CDF bound
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


ax1.set_xlim(-3, 5)
ax1.set_ylim(0, 1)
ax1.set_xlabel('Joint Objective Value (dB)')
ax1.set_ylabel('Empirical CDF')
ax1.legend(loc='best')
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



#%% Final figure generation script (Version 2)

zoom_which_alpha = 2    # first or second alpha value to zoom {1,2}
SAVE_FIGS = True
CONSOLE_RUN = True
cwd = os.getcwd()
console_run = cwd + '/src/optimization/exp1_baseline' if CONSOLE_RUN else ''
save_path = os.path.join(console_run, 'figures')
os.makedirs(save_path, exist_ok=True)

lw = 1.8
lwin = 2
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
    "milp":         '--',
    "greedy":       '--',
    "ch_only":      '--',
    "comm_only":    '--',
    "sens_only":    '--',

    "milp2":        '-',
    "greedy2":      '-',
    "ch_only2":     '-',
    "comm_only2":   '-',
    "sens_only2":   '-',
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

ax1.plot(x_milp, y_milp,    linestyle=dashes['milp'],        color=palette['milp'],      lw=lw)             # label=rf'MILP ($\alpha$={alpha})'
ax1.plot(x_grdy, y_grdy,    linestyle=dashes['greedy'],      color=palette['greedy'],    lw=lw)             # label=r'Greedy (MILP-aligned)'
ax1.plot(x_ch,   y_ch,      linestyle=dashes['ch_only'],     color=palette['ch_only'],   lw=lw)             # label='Greedy (channel-only)'
ax1.plot(x_comm+x_comm_offset, y_comm,linestyle=dashes['comm_only'],   color=palette['comm_only'], lw=lw)   # label='Greedy (comm-only)'
ax1.plot(x_sens, y_sens,    linestyle=dashes['sens_only'],   color=palette['sens_only'], lw=lw)             # label='Greedy (sens-only)'


ax1.plot(x_milp_2ndAlpha, y_milp_2ndAlpha,    linestyle=dashes['milp2'],        color=palette['milp'],      lw=lw, label=rf'MILP')
ax1.plot(x_grdy_2ndAlpha, y_grdy_2ndAlpha,    linestyle=dashes['greedy2'],      color=palette['greedy'],    lw=lw, label=rf'Grdy (MILP-aligned)')
ax1.plot(x_ch_2ndAlpha,   y_ch_2ndAlpha,      linestyle=dashes['ch_only2'],     color=palette['ch_only'],   lw=lw, label='Grdy (channel-only)')
ax1.plot(x_comm_2ndAlpha+x_comm_offset, y_comm_2ndAlpha,linestyle=dashes['comm_only2'],   color=palette['comm_only'], lw=lw, label='Grdy (comm-only)')
ax1.plot(x_sens_2ndAlpha+x_sens_offset, y_sens_2ndAlpha,    linestyle=dashes['sens_only2'],   color=palette['sens_only'], lw=lw, label='Grdy (sens-only)')




axins = zoomed_inset_axes(ax1, zoom=5.5, loc="upper right")  # try "lower right" if overlap
axins.set_facecolor("white")

# re-plot in the inset
if zoom_which_alpha == 1:
    axins.plot(x_milp, y_milp, linestyle=dashes["milp"],      color=palette["milp"],      lw=lw)
    axins.plot(x_ch,   y_ch,   linestyle=dashes["ch_only"],   color=palette["ch_only"],   lw=lw)
    axins.plot(x_comm+x_comm_offset, y_comm, linestyle=dashes["comm_only"], color=palette["comm_only"], lw=lw)
    axins.plot(x_sens, y_sens, linestyle=dashes["sens_only"], color=palette["sens_only"], lw=lw)
    axins.plot(x_grdy, y_grdy, linestyle=dashes["greedy"],    color=palette["greedy"],    lw=lw)
else:
    axins.plot(x_milp_2ndAlpha, y_milp_2ndAlpha, linestyle=dashes["milp2"],      color=palette["milp"],      lw=lwin)
    axins.plot(x_ch_2ndAlpha,   y_ch_2ndAlpha,   linestyle=dashes["ch_only2"],   color=palette["ch_only"],   lw=lwin)
    axins.plot(x_comm_2ndAlpha+x_comm_offset, y_comm_2ndAlpha, linestyle=dashes["comm_only2"], color=palette["comm_only"], lw=lwin)
    axins.plot(x_sens_2ndAlpha+x_sens_offset, y_sens_2ndAlpha, linestyle=dashes["sens_only2"], color=palette["sens_only"], lw=lwin)
    axins.plot(x_grdy_2ndAlpha, y_grdy_2ndAlpha, linestyle=dashes["greedy2"],    color=palette["greedy"],    lw=lwin)

# focus window (e.g., between medians of the 4 greedy curves)
x_left  = 2.4
x_right = 2.51

# set EXACT margins
x_margin_left  = 0.10    # e.g., 0.10 objective units
x_margin_right = 0.10
y_bottom = 0.65          # lower CDF bound
y_top    = 0.75          # upper CDF bound

axins.set_xlim(x_left - x_margin_left, x_right + x_margin_right)
axins.set_ylim(y_bottom, y_top)


axins.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.4)
# hide inset tick labels
for t in axins.get_xticklabels() + axins.get_yticklabels():
    t.set_visible(False)
axins.tick_params(axis="both", which="both", length=0)

# draw rectangle + connectors
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=1.0, alpha=0.9)


ax1.set_xlim(-3, 5)
ax1.set_ylim(0, 1)
ax1.set_xlabel('Joint Objective Value (dB)')
ax1.set_ylabel('Empirical CDF')
ax1.legend(loc='best')
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
