
import numpy as np
import pickle
import os
import src.utils.library as lib

CONSOLE_RUN = True
cwd = os.getcwd()
console_run = cwd + '/src/optimization/exp1_baseline' if CONSOLE_RUN else ''

run_id = 'run_01'
filename = '2025-10-02-16-18_results.pkl'
file_path = os.path.join(console_run, run_id, filename)

results = lib.load_results(file_path)

print(f"[LOAD] --- Loaded results from '{file_path}' with {len(results)} entries.")


#%%

for i in range(len(results)):
    print('results pos_seed: ', results[i]['pos_seed'], 'results ch_seed: ', results[i]['ch_seed'])
    print('opt_status: ', results[i]['opt_status'], 'obj_val: ', results[i]['opt_objVal'])


#%%

solutions = [entry["solution"] for entry in results]

#%%

width = 50
iters = 10000
secs = 100.53954989858
print("-" * width)
print(">> Simulation finished <<".center(width))
print(f"{'Total iterations:':<20} {iters}")
print(f"{'Execution time:':<20} {secs//60:.0f} min and {secs%60:.2f} seconds")
print("-" * width)
