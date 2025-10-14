
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

#%%

user_pos_list = [entry["user_positions"] for entry in coordinates]
