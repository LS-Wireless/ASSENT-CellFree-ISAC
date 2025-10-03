
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.utils.library as lib
import src.utils.visualization_utils as viz
import src.utils.optimization_utils as opt
import src.utils.network_utils as net


N_ap = 8
M_a = 16
N_RF = 4
N_cu = 10
N_tg = 4
netparams = net.NetworkParams(N_ap=N_ap, M_a=M_a, N_RF=N_RF, N_cu=N_cu, N_tg=N_tg)

network = net.NetworkEnvironment(netparams)
netparams.ap_position_mode = 'circle'
netparams.ap_circle_radius = 350
netparams.ap_position_random_state = 10
netparams.user_position_random_state = 10
netparams.target_position_random_state = 10

netparams.user_channel_random_state = 10
netparams.target_channel_random_state = 10


network.generate_topology()
network.plot_topology()

G_comm, S_comm = network.generate_commLink_features()
G_sens = network.generate_sensLink_features()

max_gain = max(np.max(G_comm), np.max(G_sens))
G_comm_norm = G_comm / max_gain
G_sens_norm = G_sens / max_gain

optparams = opt.ProblemParams(G_comm=G_comm_norm, G_sens=G_sens_norm, S_mat=S_comm, alpha=0.6, N_RF=N_RF,
                              K_tx=2, K_rx=2, interf_penalty=0.01, rho_thresh=0.8)
optparams.change(lambda_cu=1.0, lambda_tg=1.0)

solution = opt.solve_problem(optparams, print_status=False)

#%% Saving

import pickle

# with open(f"netparams.pkl", "wb") as f:
#     pickle.dump(netparams, f)

# with open(f"solution.pkl", "wb") as f:
#     pickle.dump(solution, f)

import os

cwd = os.getcwd()

# with open(f"netparams.pkl", "rb") as f:
#     netparams_opended = pickle.load(f)
# netparams_opended.summary()

# with open(f"solution.pkl", "rb") as f:
#     sol_opended = pickle.load(f)
# sol_opended.summary()


save_path = cwd + '/src/optimization'
lib.save_dataclass_hybrid(netparams, save_path, filename='netparams')


# loading

arrays_dict = np.load(f"{save_path}/netparams_arrays.npz")

#%%

from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
print(timestamp)

#%% Visualization

viz.print_solution_summary(solution)

# System Snapshot
viz.plot_system_snapshot(ap_pos=network.ap_positions, user_pos=network.user_positions, target_pos=network.target_positions, sol=solution)

# Resource Utilization Plot
ap_names = [f'AP {i}' for i in range(N_ap)]
# For Tx APs, streams are comm+sens, for Rx APs they are 0
comm = solution.x.sum(axis=1)
sens = solution.y_tx.sum(axis=1)

viz.plot_ap_utilization(ap_names, comm, sens, is_rx=(1 - solution.tau), n_rf=N_RF)
