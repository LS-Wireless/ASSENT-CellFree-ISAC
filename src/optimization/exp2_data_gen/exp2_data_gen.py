
# exp2_data_gen.py:
# Reading {G_comm, S_comm, G_sens} data from exp1_baseline/run_03 and generating training data by solving the MILP

import os
import json
import numpy as np
import src.utils.optimization_utils as opt
import src.utils.library as lib
import time
from datetime import datetime
import sys

run_id = 'run_03'
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M")
log_path = os.path.join(run_id, f"{timestamp}_log_{run_id}.txt")
sys.stdout = lib.TeeLogger(log_path)

start_time = time.time()
width = 50
print("=" * width)
print(">> Simulation for 'exp2_data_gen' Started <<".center(width))
print(f'{now.strftime("%A, %B %d, %Y at %I:%M:%S %p")}')
print("=" * width)

# Load metadata
experiment = 'exp2_data_gen'
save_path = os.path.join('./', run_id)
if not os.path.exists(save_path):
    raise FileNotFoundError(f"Save path '{save_path}' does not exist.")
metadata_path = os.path.join(save_path, 'metadata.json')
with open(metadata_path) as f:
    metadata = json.load(f)
lib.print_log(tag='LOAD', message=f"Loaded current (exp2_data_gen) metadata from '{metadata_path}'")

config = metadata['config']
# Check
if config['run_id'] != run_id:
    raise ValueError(f"Run ID in metadata.json ({config['run_id']}) does not match run ID in function call ({run_id}).")
if config['experiment'] != experiment:
    raise ValueError(f"Experiment in metadata.json ({config['experiment']}) does not match experiment in function call ({experiment}).")

# Load input data
input_folder_path = config['path_to_input_data']
if not os.path.exists(input_folder_path):
    raise FileNotFoundError(f"Folder path '{input_folder_path}' does not exist.")
input_metadata_path = os.path.join(input_folder_path, 'metadata.json')
with open(input_metadata_path) as f:
    input_metadata = json.load(f)
lib.print_log(tag='LOAD', message=f"Loaded input (exp1_baseline) metadata from '{input_metadata_path}'")
lib.print_log(tag='INFO', message=f"\"Results from exp1 will be used as input for exp2\"")
num_parts = input_metadata['config']['num_parts_to_save']
input_file_name = config['input_file_name']

num_parts_to_load = config.get('input_parts_to_load', num_parts)
lib.print_log(tag='CONFIG', message=f"Number of files to load: {num_parts_to_load} out of {num_parts} input data files")

if num_parts == 1:
    filename = f'{input_file_name}.pkl'
    file_path = os.path.join(input_folder_path, filename)
    results = lib.load_results(file_path)
    lib.print_log(tag='LOAD', message=f"Loading file '{file_path}'")
else:
    results = []
    for i in range(num_parts_to_load):
        filename = f'{input_file_name}_p{i+1}_of_{num_parts}.pkl'
        file_path = os.path.join(input_folder_path, filename)
        results += lib.load_results(file_path)
        lib.print_log(tag='LOAD', message=f"Loading file '{file_path}'")
input_nsamps = len(results)
lib.print_log(tag='LOAD', message=f"Loaded results from '{input_folder_path}' with {input_nsamps} entries")

G_comm_list = [entry["G_comm"] for entry in results]
S_comm_list = [entry["S_comm"] for entry in results]
G_sens_list = [entry["G_sens"] for entry in results]

SAVE_RESULTS = config['SAVE_RESULTS']
lib.print_log(tag='CONFIG', message=f"SAVE_RESULTS is: {SAVE_RESULTS}")
master_seed = config['master_seed']
np.random.seed(master_seed)

lib.print_log(tag='CONFIG', message=f"Running experiment '{experiment}' with run ID '{run_id}'")
result_keys = config['_full_list_of_results'] if config['results_to_save'] == "all" else config['results_to_save']
lib.print_log(tag='CONFIG', message=f"Saving results: {result_keys}")

lib.print_log(tag='CONFIG', message=f"Range of alpha: {metadata['config']['alpha_range']}")
lib.print_log(tag='CONFIG', message=f"Range of lambda_cu: {metadata['config']['lambda_cu_range']}")
lib.print_log(tag='CONFIG', message=f"Range of lambda_tg: {metadata['config']['lambda_tg_range']}")

add_reward = config.get('add_reward', True)
lib.print_log(tag='CONFIG', message=f"add_reward for optimization: {add_reward}")

total_iters = len(results)
lib.print_log(tag='RUN', message=f"Total number of iterations: {total_iters}")

num_parts_to_save = config['num_parts_to_save']
if num_parts_to_save > 1:
    q = total_iters // num_parts_to_save
    r = total_iters % num_parts_to_save
    save_indices = np.arange(q, q * num_parts_to_save + q, q)
    save_indices[-1] += r
else:
    save_indices = None

saver = lib.ResultSaver()
optparams = None
save_part_counter = 1
lib.print_log(tag='RUN', message=f"Starting iterations ...\n")
for it in range(total_iters):
    if (it + 1) % 50 == 0:
        print(f"[RUN] Running for iteration {it + 1}/{total_iters} ...")

    G_comm = G_comm_list[it]
    S_comm = S_comm_list[it]
    G_sens = G_sens_list[it]
    max_gain = max(np.max(G_comm), np.max(G_sens))
    G_comm_norm = G_comm / max_gain
    G_sens_norm = G_sens / max_gain

    alpha = np.random.uniform(low=config['alpha_range'][0], high=config['alpha_range'][1])
    lambda_cu = np.random.uniform(low=config['lambda_cu_range'][0], high=config['lambda_cu_range'][1], size=input_metadata['NetworkParams']['N_cu'])
    lambda_tg = np.random.uniform(low=config['lambda_tg_range'][0], high=config['lambda_tg_range'][1], size=input_metadata['NetworkParams']['N_tg'])

    if it == 0:
        optparams = opt.ProblemParams(G_comm=G_comm_norm, G_sens=G_sens_norm, S_mat=S_comm, alpha=alpha,
                                      N_RF=input_metadata['NetworkParams']['N_RF'], K_tx=config['K_tx'], K_rx=config['K_rx'],
                                      interf_penalty=config['interf_penalty'], rho_thresh=config['rho_thresh'],
                                      lambda_cu=lambda_cu, lambda_tg=lambda_tg)
    else:
        optparams.change(G_comm=G_comm_norm, G_sens=G_sens_norm, S_mat=S_comm, alpha=alpha,
                         lambda_cu=lambda_cu, lambda_tg=lambda_tg, update_dependencies=True)

    solution, opt_status, opt_objVal = opt.solve_problem(optparams, return_status=True, return_objVal=True,
                                                         print_status=False, add_reward=add_reward)

    loop_result = {'alpha': alpha, 'lambda_cu': lambda_cu, 'lambda_tg': lambda_tg,
                   'opt_status': opt_status, 'opt_objVal': opt_objVal, 'solution': solution}
    filtered_result = {k: loop_result[k] for k in result_keys if k in loop_result}
    saver.add(filtered_result)
    if SAVE_RESULTS and save_indices is not None and (it+1) in save_indices:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        reward_tag = '' if add_reward else '_noReward'
        file_path = os.path.join(save_path, timestamp + f'_results_p{save_part_counter}_of_{num_parts_to_save}{reward_tag}.pkl')
        saver.save(file_path)
        lib.print_log(tag='SAVE', message=f"Saved part {save_part_counter} out of {num_parts_to_save} to: '{file_path}'\n")
        saver.reset()
        save_part_counter += 1


lib.print_log(tag='RUN', message=f"Finished iterations.\n")

if SAVE_RESULTS and save_indices is None:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    reward_tag = '' if add_reward else '_noReward'
    file_path = os.path.join(save_path, timestamp + f'_results{reward_tag}.pkl')
    saver.save(file_path)
    lib.print_log(tag='SAVE', message=f"Saved results to: '{file_path}'")

end_time = time.time()
duration = end_time - start_time
print("=" * width)
print(">> Simulation for 'exp2_data_gen' finished <<".center(width))
print(f"{'Total iterations:':<20} {total_iters}")
print(f"{'Execution time:':<20} {duration//60:.0f} min and {duration%60:.2f} seconds")
print("=" * width)
