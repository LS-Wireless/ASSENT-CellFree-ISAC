
import numpy as np
import os
import json
import src.utils.library as lib
import matplotlib.pyplot as plt

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

lib.print_log(tag='LOAD', message=f"Loaded results from '{folder_path}' with {len(results)} entries")

#%%

G_comm_list = [entry["G_comm"] for entry in results]
S_comm_list = [entry["S_comm"] for entry in results]
G_sens_list = [entry["G_sens"] for entry in results]
solution_list = [entry["solution"] for entry in results]

#%%

coordinates_path = os.path.join(folder_path, 'coordinates.pkl')
coordinates = lib.load_results(coordinates_path)

user_pos_list = [entry["user_positions"] for entry in coordinates]

#%%

G_comm = G_comm_list[0]
G_sens = G_sens_list[0]
max_gain = max(np.max(G_comm), np.max(G_sens))
G_comm_norm = G_comm / max_gain
G_sens_norm = G_sens / max_gain

print(f"G_comm_norm min: {np.min(G_comm_norm)}, avg: {np.mean(G_comm_norm)}, max: {np.max(G_comm_norm)}")
print(f"G_sens_norm min: {np.min(G_sens_norm)}, avg: {np.mean(G_sens_norm)}, max: {np.max(G_sens_norm)}")

G_comm_db = 10 * np.log10(G_comm_norm)
G_sens_db = 10 * np.log10(G_sens_norm)

print(f"G_comm_db min: {np.min(G_comm_db)}, avg: {np.mean(G_comm_db)}, max: {np.max(G_comm_db)}")
print(f"G_sens_db min: {np.min(G_sens_db)}, avg: {np.mean(G_sens_db)}, max: {np.max(G_sens_db)}")

G_comm_stand = (G_comm_db - np.mean(G_comm_db)) / np.std(G_comm_db)
G_sens_stand = (G_sens_db - np.mean(G_sens_db)) / np.std(G_sens_db)

print(f"G_comm_stand min: {np.min(G_comm_stand)}, avg: {np.mean(G_comm_stand)}, max: {np.max(G_comm_stand)}")
print(f"G_sens_stand min: {np.min(G_sens_stand)}, avg: {np.mean(G_sens_stand)}, max: {np.max(G_sens_stand)}")
