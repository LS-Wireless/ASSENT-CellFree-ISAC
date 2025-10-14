
import numpy as np
import os
import json
import src.utils.library as lib
import src.utils.network_utils as net


SAVE_RESULTS = True
CONSOLE_RUN = False

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

lib.print_log(tag='LOAD', message=f"Loaded results from '{folder_path}' with {len(results)} entries")
pos_seed_list = [entry["pos_seed"] for entry in results]
unique_seeds = list(dict.fromkeys(pos_seed_list))



# Network Parameters
netparams_dict = metadata['NetworkParams']
netparams = net.NetworkParams(**netparams_dict)

network = net.NetworkEnvironment(netparams)

saver = lib.ResultSaver()
for it in range(len(unique_seeds)):
    netparams.user_position_random_state = unique_seeds[it]
    netparams.target_position_random_state = unique_seeds[it]
    network.generate_topology()
    ap_positions = network.ap_positions
    user_positions = network.user_positions
    target_positions = network.target_positions
    coordinates = {'ap_positions': ap_positions, 'user_positions': user_positions, 'target_positions': target_positions, 'num_position_realizations': len(unique_seeds)}
    saver.add(coordinates)



if SAVE_RESULTS:
    file_path = os.path.join(folder_path, 'coordinates.pkl')
    saver.save(file_path)
    lib.print_log(tag='SAVE', message=f"Saved results to: '{file_path}'")
