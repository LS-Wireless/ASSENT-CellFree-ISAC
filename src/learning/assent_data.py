
import torch
import src.utils.learning_utils as lu
import os
import json
import src.utils.library as lib

CONSOLE_RUN = False
cwd = os.getcwd()
save_path = cwd + '/src/learning' if CONSOLE_RUN else './'
console_run = cwd + '/src/optimization/exp1_baseline' if CONSOLE_RUN else '../optimization/exp1_baseline'

exp1_run_id = 'run_01'
exp1_folder_path = os.path.join(console_run, exp1_run_id)
metadata_path = os.path.join(exp1_folder_path, 'metadata.json')
with open(metadata_path) as f:
    exp1_metadata = json.load(f)
num_parts = exp1_metadata['config']['num_parts_to_save']
N_RF = exp1_metadata['NetworkParams']['N_RF']
M_a = exp1_metadata['NetworkParams']['M_a']
env_dim = max(exp1_metadata['NetworkParams']['env_x'], exp1_metadata['NetworkParams']['env_y'])

exp1_results = []
for i in range(num_parts):
    filename = f'2025-10-03_results_p{i+1}_of_{num_parts}.pkl'
    file_path = os.path.join(exp1_folder_path, filename)
    exp1_results += lib.load_results(file_path)
coordinates_path = os.path.join(exp1_folder_path, 'coordinates.pkl')
coordinates = lib.load_results(coordinates_path)
num_pos_reals = coordinates[0]['num_position_realizations']
lib.print_log(tag='LOAD', message=f"Loaded exp1 results and coordinates from '{exp1_folder_path}'")

console_run = cwd + '/src/optimization/exp2_data_gen' if CONSOLE_RUN else '../optimization/exp2_data_gen'
exp2_run_id = 'run_01'
exp2_folder_path = os.path.join(console_run, exp2_run_id)
file_path = os.path.join(exp2_folder_path, '2025-10-13-13-25_results.pkl')
exp2_results = lib.load_results(file_path)
lib.print_log(tag='LOAD', message=f"Loaded exp2 results from '{exp2_folder_path}'")

G_comm_list = [entry["G_comm"] for entry in exp1_results]
S_comm_list = [entry["S_comm"] for entry in exp1_results]
G_sens_list = [entry["G_sens"] for entry in exp1_results]

alpha_list = [entry["alpha"] for entry in exp2_results]
lambda_cu_list = [entry["lambda_cu"] for entry in exp2_results]
lambda_tg_list = [entry["lambda_tg"] for entry in exp2_results]
solution_list = [entry["solution"] for entry in exp2_results]



total_iters = len(exp2_results)
full_dataset = []
for idx in range(total_iters):
    if (idx+1) % 100 == 0:
        lib.print_log(tag='RUN', message=f"Running for iteration {idx+1}/{total_iters} ...")
    snapshot_dict = lu.prepare_snapshot_data(coordinates[idx//num_pos_reals], G_comm_list[idx], S_comm_list[idx], G_sens_list[idx],
                                             alpha_list[idx], lambda_cu_list[idx], lambda_tg_list[idx], solution_list[idx],
                                             N_RF, M_a, env_dim=env_dim, normalize=True)
    graph_sample = lu.create_graph_sample(snapshot_dict)
    full_dataset.append(graph_sample)
file_path = os.path.join(save_path, 'final_graph_dataset.pt')
torch.save(full_dataset, file_path)
lib.print_log(tag='SAVE', message=f"Saved final graph dataset to '{file_path}'")


