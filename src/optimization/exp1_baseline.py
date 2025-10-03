
import os
import json
import numpy as np
import src.utils.optimization_utils as opt
import src.utils.network_utils as net
import src.utils.library as lib
import time
from datetime import datetime

start_time = time.time()
width = 50
print("=" * width)
print(">> Simulation Started <<".center(width))
print("=" * width)

# Load metadata
run_id = 'run_01'
experiment = 'exp1_baseline'
save_path = os.path.join(experiment, run_id)
if not os.path.exists(save_path):
    raise FileNotFoundError(f"Save path '{save_path}' does not exist.")

metadata_path = os.path.join(save_path, 'metadata.json')
with open(metadata_path) as f:
    metadata = json.load(f)
lib.print_log(tag='LOAD', message=f"Loaded metadata from '{metadata_path}'")

config = metadata['config']
# Check
if config['run_id'] != run_id:
    raise ValueError(f"Run ID in metadata.json ({config['run_id']}) does not match run ID in function call ({run_id}).")
if config['experiment'] != experiment:
    raise ValueError(f"Experiment in metadata.json ({config['experiment']}) does not match experiment in function call ({experiment}).")

SAVE_RESULTS = config['SAVE_RESULTS']
master_seed = config['master_seed']
np.random.seed(master_seed)
num_entity_position_realizations = config['num_entity_position_realizations']
num_entity_channel_realizations = config['num_entity_channel_realizations']
sample_size = int(max(num_entity_position_realizations, num_entity_channel_realizations) * 10)

entity_position_seeds = np.random.choice(sample_size, num_entity_position_realizations, replace=False)
entity_channel_seeds = np.random.choice(sample_size, num_entity_channel_realizations, replace=False)

lib.print_log(tag='CONFIG', message=f"Running experiment '{experiment}' with run ID '{run_id}'")
result_keys = config['_full_list_of_results'] if config['results_to_save'] == "all" else config['results_to_save']
lib.print_log(tag='CONFIG', message=f"Saving results: {result_keys}")

total_iters = num_entity_position_realizations * num_entity_channel_realizations
lib.print_log(tag='RUN', message=f"Total number of iterations: {total_iters}")

# Network Parameters
netparams_dict = metadata['NetworkParams']
netparams = net.NetworkParams(**netparams_dict)

network = net.NetworkEnvironment(netparams)

saver = lib.ResultSaver()
optparams = None
lib.print_log(tag='RUN', message=f"Starting iterations ...\n")
for pos_iter in range(num_entity_position_realizations):
    netparams.user_position_random_state = entity_position_seeds[pos_iter]
    netparams.target_position_random_state = entity_position_seeds[pos_iter]

    for ch_iter in range(num_entity_channel_realizations):
        netparams.user_channel_random_state = entity_channel_seeds[ch_iter]
        netparams.target_channel_random_state = entity_channel_seeds[ch_iter]
        if (ch_iter+1) % 5 == 0:
            print(f"[RUN] Running position realization {pos_iter+1}/{num_entity_position_realizations} and channel realization {ch_iter+1}/{num_entity_channel_realizations} ...")

        network.generate_topology()
        G_comm, S_comm = network.generate_commLink_features()
        G_sens = network.generate_sensLink_features()

        max_gain = max(np.max(G_comm), np.max(G_sens))
        G_comm_norm = G_comm / max_gain
        G_sens_norm = G_sens / max_gain

        if pos_iter == 0 and ch_iter == 0:
            optparams = opt.ProblemParams(G_comm=G_comm_norm, G_sens=G_sens_norm, S_mat=S_comm, alpha=config['alpha'],
                                          N_RF=netparams.N_RF, K_tx=config['K_tx'], K_rx=config['K_rx'],
                                          interf_penalty=config['interf_penalty'], rho_thresh=config['rho_thresh'],
                                          lambda_cu=config['lambda_cu'], lambda_tg=config['lambda_tg'])
        else:
            optparams.change(G_comm=G_comm_norm, G_sens=G_sens_norm, S_mat=S_comm, update_dependencies=True)

        solution, opt_status, opt_objVal = opt.solve_problem(optparams, return_status=True, return_objVal=True, print_status=False)

        loop_result = {'pos_seed': entity_position_seeds[pos_iter], 'ch_seed': entity_channel_seeds[ch_iter],
                       'G_comm': G_comm, 'S_comm': S_comm, 'G_sens': G_sens,
                       'opt_status': opt_status, 'opt_objVal': opt_objVal, 'solution': solution}
        filtered_result = {k: loop_result[k] for k in result_keys if k in loop_result}
        saver.add(filtered_result)
lib.print_log(tag='RUN', message=f"Finished iterations.\n")

if SAVE_RESULTS:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    file_path = os.path.join(save_path, timestamp + '_results.pkl')
    saver.save(file_path)
    lib.print_log(tag='SAVE', message=f"Saved results to: '{file_path}'")

end_time = time.time()
duration = end_time - start_time
print("=" * width)
print(">> Simulation finished <<".center(width))
print(f"{'Total iterations:':<20} {total_iters}")
print(f"{'Execution time:':<20} {duration//60:.0f} min and {duration%60:.2f} seconds")
print("=" * width)

