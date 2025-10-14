
import numpy as np
import os
import json
import src.utils.library as lib
import matplotlib.pyplot as plt


CONSOLE_RUN = True
cwd = os.getcwd()
console_run = cwd + '/src/optimization/exp2_data_gen' if CONSOLE_RUN else ''

run_id = 'run_01'
folder_path = os.path.join(console_run, run_id)
metadata_path = os.path.join(folder_path, 'metadata.json')
with open(metadata_path) as f:
    metadata = json.load(f)

filename = f'2025-10-13-13-25_results.pkl'
file_path = os.path.join(folder_path, filename)
results = lib.load_results(file_path)

lib.print_log(tag='LOAD', message=f"Loaded results from '{folder_path}' with {len(results)} entries")

#%%

alpha_list = [entry["alpha"] for entry in results]
lambda_cu_list = [entry["lambda_cu"] for entry in results]
lambda_tg_list = [entry["lambda_tg"] for entry in results]
opt_status_list = [entry["opt_status"] for entry in results]
opt_objVal_list = [entry["opt_objVal"] for entry in results]

#%%

plt.hist(opt_objVal_list, bins=500, density=True)
plt.show()
