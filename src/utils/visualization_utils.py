
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import src.utils.optimization_utils as opt
import seaborn as sns


# >> Function to plot topology
def plot_topology(ap_positions, user_positions, target_positions, env_x, env_y, show='all', figsize=(8, 7), margin=10, equal_aspect=False):
    """
    Plot the topology of the cell-free ISAC network.
    :param ap_positions: AP positions
    :param user_positions: User positions
    :param target_positions: Target positions
    :param env_x: Env x dimension
    :param env_y: Env y dimension
    :param show: What to show ('all', 'AP', 'CU', 'T', 'AP-CU', 'AP-T', 'CU-T')
    :param figsize: The figsize
    :param margin: The margin for the box around the environment
    :param equal_aspect: If True, set aspect ratio to 1:1 (equal aspect ratio)
    :return: No return value
    """
    plt.rcdefaults()
    show_modes = ['all', 'AP', 'CU', 'T', 'AP-CU', 'AP-T', 'CU-T']

    plt.rcParams.update({
        "font.size": 14,  # default text size
        "axes.titlesize": 16,  # title
        "axes.labelsize": 14,  # x and y labels
        "xtick.labelsize": 12,  # x tick labels
        "ytick.labelsize": 12,  # y tick labels
        "legend.fontsize": 12  # legend
    })
    annotate_font = 11
    fig, ax = plt.subplots(figsize=figsize)
    # Draw environment rectangle with light background
    rect = patches.Rectangle((0, 0), env_x, env_y, linewidth=2, linestyle='--', edgecolor='black', facecolor='#f0f0f0', alpha=0.9,
                             zorder=0)
    ax.add_patch(rect)

    # --- Plot APs ---
    if show == 'all' or show == 'AP' or show == 'AP-CU' or show == 'AP-T':
        ax.scatter(ap_positions[:, 0], ap_positions[:, 1], c='red', marker='^', s=220, edgecolors='k', linewidth=1.2,
                   label='AP')
        for i, (x, y) in enumerate(ap_positions):
            ax.annotate(f'AP{i}', (x, y), xytext=(4, 4), textcoords='offset points', fontsize=annotate_font, color='darkred')

    # --- Plot Users ---
    if show == 'all' or show == 'CU' or show == 'AP-CU' or show == 'CU-T':
        ax.scatter(user_positions[:, 0], user_positions[:, 1], c='royalblue', marker='o', s=120, edgecolors='k',
                   linewidth=0.8, label='Comm User')
        for i, (x, y) in enumerate(user_positions):
            ax.annotate(f'CU{i}', (x, y), xytext=(4, 4), textcoords='offset points', fontsize=annotate_font, color='navy')

    # --- Plot Targets ---
    if show == 'all' or show == 'T' or show == 'AP-T' or show == 'CU-T':
        ax.scatter(target_positions[:, 0], target_positions[:, 1], c='green', marker='X', s=180, edgecolors='k',
                   linewidth=1.2, label='Target')
        for i, (x, y) in enumerate(target_positions):
            ax.annotate(f'T{i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=annotate_font, color='darkgreen')

    if show not in show_modes:
        raise ValueError(f"Invalid show mode. Choose from {show_modes}.")
    # --- Formatting ---
    if equal_aspect:
        ax.set_xlim([-margin, env_x + margin])
        ax.set_ylim([-margin, env_y + margin])
        ax.set_aspect('equal', adjustable='box')
    else:
        env_margin = np.max([env_x, env_y]) / 2 + margin
        ax.set_xlim([env_x / 2 - env_margin, env_x / 2 + env_margin])
        ax.set_ylim([env_y / 2 - env_margin, env_y / 2 + env_margin])


    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Cell-Free ISAC Network Topology")

    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    legend = ax.legend(ncol=3, loc='upper center', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_alpha(1)

    ax.ticklabel_format(axis='both', style='sci', scilimits=(1, 2))
    plt.tight_layout()
    plt.show()



def plot_commChannel_gains(G_mat, dB_scale=True, normalize2max=False, cmap='viridis', figsize=(6,5), ap_indices=None, user_indices=None):
    """
    Plots a 2D heatmap of the channel gains for given APs and users.
    :param G_mat: Full matrix of channel gains (N_ap x N_cu)
    :param dB_scale: Whether to show the channel gains in dB (default: True)
    :param normalize2max: Whether to normalize the channel gains to the maximum value (default: False)
    :param cmap: Figure colormap (default: 'viridis')
    :param figsize: Figure size (default: (6, 5)))
    :param ap_indices: In case of specific AP indices (default: None)
    :param user_indices: In case of specific user indices (default: None)
    :return: No return
    """
    plt.rcdefaults()
    G_mat = G_mat / np.max(G_mat) if normalize2max else G_mat
    mat = 10 * np.log10(G_mat) if dB_scale else G_mat
    txt_fmt = '.1f' if dB_scale else '.3f'
    txt_title = '(dB)' if dB_scale else '(Linear)'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect='auto', cmap=cmap)
    plt.colorbar(im, ax=ax)

    # Annotate each cell
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], txt_fmt), ha='center', va='center',
                    color='white' if mat[i, j] < 0.5 else 'black', fontsize=10)

    ax.set_title(f"AP-User Channel Gains {txt_title}")
    if ap_indices is not None:
        ax.set_yticks(np.arange(mat.shape[0]), ap_indices)
    else:
        ax.set_yticks(np.arange(mat.shape[0]))
    if user_indices is not None:
        ax.set_xticks(np.arange(mat.shape[1]), user_indices)
    else:
        ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xlabel("User Index")
    ax.set_ylabel("AP Index")
    plt.tight_layout()
    plt.show()


def plot_commChannel_correlations(S_mat_2D, cmap='viridis', figsize=(6,5), ap_index=None, user_indices=None):
    """
    Plots a 2D heatmap of the spatial correlations for given users.
    :param S_mat_2D: Correlation matrix (N_cu x N_cu)
    :param cmap: Colormap (default: 'viridis')
    :param figsize: Figure size (default: (6, 5)))
    :param ap_index: AP index (default: None)
    :param user_indices: In case of specific user indices (default: None)
    :return: No return
    """
    plt.rcdefaults()
    mat = S_mat_2D
    n = mat.shape[0]
    txt_title = f'(AP {ap_index})' if ap_index is not None else ''

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center',
                    color='white' if mat[i, j] < 0.5 else 'black', fontsize=10)

    if user_indices is not None:
        ax.set_xticks(np.arange(n), user_indices)
        ax.set_yticks(np.arange(n), user_indices)
    ax.set_title(f"User-User Correlation Matrix {txt_title}")
    ax.set_xlabel("User Index")
    ax.set_ylabel("User Index")
    plt.tight_layout()
    plt.show()



def plot_sensing_channel_gains(G_tg_mat_2D, target_id=None, cmap='viridis', figsize=(6,5), dB_scale=True, normalize2max=False, ap_indices=None):
    """
    Plots a 2D heatmap of the channel gains for given APs and targets.
    :param G_tg_mat_2D: AP-Target channel gains matrix (N_ap x N_tg)
    :param target_id: Target ID
    :param cmap: Colormap
    :param figsize: Figure size
    :param dB_scale: Whether to show the channel gains in dB
    :param normalize2max: Whether to normalize the channel gains to the maximum value
    :param ap_indices: In case of specific AP indices
    :return: No return
    """
    plt.rcdefaults()
    G_tg_mat_2D = G_tg_mat_2D / np.max(G_tg_mat_2D) if normalize2max else G_tg_mat_2D
    mat = 10 * np.log10(G_tg_mat_2D) if dB_scale else G_tg_mat_2D
    txt_fmt = '.1f' if dB_scale else '.3f'
    txt_title = '(dB)' if dB_scale else '(Linear)'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect='auto', cmap=cmap)
    plt.colorbar(im, ax=ax)

    # Annotate each cell
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], txt_fmt), ha='center', va='center',
                    color='white' if mat[i, j] < 0.5 else 'black', fontsize=8)

    ax.set_title(f"AP-Target Channel Gains {txt_title}" + (f" for Target {target_id}" if target_id is not None else ""))
    if ap_indices is not None:
        ax.set_xticks(np.arange(mat.shape[0]), ap_indices)
        ax.set_yticks(np.arange(mat.shape[0]), ap_indices)
    else:
        ax.set_xticks(np.arange(mat.shape[0]))
        ax.set_yticks(np.arange(mat.shape[0]))

    ax.set_xlabel("AP Index (Tx/Rx)")
    ax.set_ylabel("AP Index (Tx/Rx)")
    plt.tight_layout()
    plt.show()




def print_solution_summary(sol: opt.Solution):
    """
    Prints a summary of the optimization solution.
    :param sol: Solution dataclass defined in optimization_utils.py
    :return: No return
    """
    N_ap = sol.x.shape[0]
    N_cu = sol.x.shape[1]
    N_tg = sol.y_tx.shape[1]
    print("\n--- Solution Summary ---")
    print("\nCommunication Associations (AP → Users):")
    for a in range(N_ap):
        users_served = [u for u in range(N_cu) if sol.x[a, u] > 0.5]
        if users_served:
            print(f"- AP {a}: Users {users_served}")

    print("\nSensing Illuminators (AP → Targets):")
    for a in range(N_ap):
        targets_tx = [t for t in range(N_tg) if sol.y_tx[a, t] > 0.5]
        if targets_tx:
            print(f"- AP {a} (Tx): Targets {targets_tx}")

    print("\nSensing Receivers (AP ← Targets):")
    for a in range(N_ap):
        targets_rx = [t for t in range(N_tg) if sol.y_rx[a, t] > 0.5]
        if targets_rx:
            print(f"- AP {a} (Rx): Targets {targets_rx}")

    print("\nScheduled Targets:")
    for t in range(N_tg):
        if sol.s[t] > 0.5:
            print(f"- Target {t} is scheduled")

    print("\nAP Mode Assignments:")
    for a in range(N_ap):
        mode = "Tx" if sol.tau[a] > 0.5 else "Rx"
        print(f"- AP {a}: {mode} mode")




def plot_system_snapshot(ap_pos, user_pos, target_pos, sol: opt.Solution):
    """
    Generates a geographic plot of a single ISAC network snapshot.
    :param ap_pos: AP positions (N_ap x 2)
    :param user_pos: User positions (N_cu x 2)
    :param target_pos: Target positions (N_tg x 2)
    :param sol: Solution dataclass defined in optimization_utils.py
    :return: No return
    """
    plt.rcdefaults()
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot APs based on their mode (tau)
    tx_aps = ap_pos[sol.tau == 1]
    rx_aps = ap_pos[sol.tau == 0]
    ax.scatter(tx_aps[:, 0], tx_aps[:, 1], c='blue', marker='^', s=150, label='Tx APs', zorder=5)
    ax.scatter(rx_aps[:, 0], rx_aps[:, 1], c='red', marker='v', s=150, label='Rx APs', zorder=5)

    # Plot users and targets
    ax.scatter(user_pos[:, 0], user_pos[:, 1], c='green', marker='o', s=50, label='Users')

    # Differentiate between scheduled and unscheduled targets
    scheduled_targets = target_pos[sol.s == 1]
    unscheduled_targets = target_pos[sol.s == 0]
    ax.scatter(scheduled_targets[:, 0], scheduled_targets[:, 1], c='purple', marker='X', s=100,
               label='Scheduled Targets')
    ax.scatter(unscheduled_targets[:, 0], unscheduled_targets[:, 1], c='gray', marker='x', s=50,
               label='Unscheduled Targets')

    # Draw communication links (x[a,u] = 1)
    for u_idx in range(user_pos.shape[0]):
        for a_idx in range(ap_pos.shape[0]):
            if sol.x[a_idx, u_idx] == 1:
                ax.plot([ap_pos[a_idx, 0], user_pos[u_idx, 0]],
                        [ap_pos[a_idx, 1], user_pos[u_idx, 1]], 'g--', lw=1)

    # Draw sensing links (y_tx and y_rx)
    for t_idx in range(target_pos.shape[0]):
        if sol.s[t_idx] == 1:
            for at_idx in range(ap_pos.shape[0]):
                if sol.y_tx[at_idx, t_idx] == 1:
                    ax.plot([ap_pos[at_idx, 0], target_pos[t_idx, 0]],
                            [ap_pos[at_idx, 1], target_pos[t_idx, 1]], 'm-.', lw=1.5,
                            label='Sensing Tx Path' if t_idx == 0 and at_idx == 0 else "")
                if sol.y_rx[at_idx, t_idx] == 1:
                    ax.plot([target_pos[t_idx, 0], ap_pos[at_idx, 0]],
                            [target_pos[t_idx, 1], ap_pos[at_idx, 1]], 'c:', lw=2,
                            label='Sensing Rx Path' if t_idx == 0 and at_idx == 0 else "")

    ax.set_title('ISAC System Snapshot', fontsize=16)
    ax.set_xlabel('X-coordinate (m)')
    ax.set_ylabel('Y-coordinate (m)')
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()




def plot_pareto_frontier(alpha_values, comm_utilities, sens_utilities):
    """
    Plots the Pareto frontier for communication and sensing utilities.
    :param alpha_values: Values of alpha used in the optimization
    :param comm_utilities: Communication utilities
    :param sens_utilities: Sensing utilities
    :return: No return
    """
    plt.rcdefaults()
    plt.figure(figsize=(8, 6))
    plt.plot(comm_utilities, sens_utilities, 'o-', c='purple', lw=2, markersize=8)
    plt.title('Communication-Sensing Pareto Frontier', fontsize=16)
    plt.xlabel('Total Communication Utility (Normalized)', fontsize=12)
    plt.ylabel('Total Sensing Utility (Normalized)', fontsize=12)
    plt.grid(True)
    # Annotate points
    for i, alpha in enumerate(alpha_values):
        plt.annotate(f'α={alpha:.1f}', (comm_utilities[i], sens_utilities[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')
    plt.show()



def plot_ap_utilization(ap_labels, comm_streams, sens_streams, is_rx, n_rf):
    """
    Plots the AP resource utilization.
    :param ap_labels: List of AP labels (e.g., 'AP1', 'AP2', etc.)
    :param comm_streams: Number of communication streams per AP
    :param sens_streams: Number of sensing streams per AP
    :param is_rx: Whether the AP is in RX mode or not (boolean array)
    :param n_rf: Number of RF chains per AP
    :return: No return
    """
    N_ap = len(ap_labels)
    if np.isscalar(n_rf):
        n_rf = np.full(N_ap, n_rf, dtype=int)
    plt.rcdefaults()
    data = np.vstack([comm_streams, sens_streams]).T
    total_streams = comm_streams + sens_streams

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".0f", cmap="viridis",
                yticklabels=ap_labels, xticklabels=['Comm Streams', 'Sensing Streams'], ax=ax)

    # Add text for Rx mode and total utilization
    for i in range(len(ap_labels)):
        util_text = f'N_RF = {n_rf[i]}'
        # ax.text(0.9, i + 0.5, util_text, ha="center", va="center", color="blue", fontsize=12)
        if is_rx[i]:
            ax.text(2.05, i + 0.5, "RX", ha="center", va="center", color="red", fontweight='bold', fontsize=12)

    ax.set_title(f'AP Resource Utilization ({n_rf[0]} RF Chains)', fontsize=16)
    plt.xticks(rotation=0)
    # ax.text(2.35, -0.5, "Total Util.  Mode", ha="center", va="center", fontstyle='italic')
    plt.tight_layout()
    plt.show()
