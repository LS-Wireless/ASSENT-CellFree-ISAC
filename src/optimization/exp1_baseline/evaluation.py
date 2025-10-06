
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

#%%

import numpy as np

for i in range(10):
    for j in range(10):
        current_iter = i * 10 + j + 1
        # print(f"Iteration {current_iter}")

total_nums = 100
divide_to = 5
q = total_nums // divide_to
r = total_nums % divide_to

nums = np.arange(q, q*divide_to+q, q)
nums[-1] += r
print(nums)

for i in range(10):
    for j in range(10):
        current_iter = i * 10 + j + 1
        if nums is not None and current_iter in nums:
            print(f"Iteration {current_iter} --> can save")
            nums = None

total_iters = 100
num_parts_to_save = 4
q = total_iters // num_parts_to_save
r = total_iters % num_parts_to_save
save_indices = np.arange(q, q * num_parts_to_save + q, q)
save_indices[-1] += r
print(save_indices)

#%%

import numpy as np
pos_iters = 10
ch_iters = 10
alpha_list = [0.1]

total_iters = len(alpha_list) * pos_iters * ch_iters

for alpha_idx, alpha_val in enumerate(alpha_list):
    for pos_iter in range(pos_iters):
        for ch_iter in range(ch_iters):
            current_iter = alpha_idx * pos_iters * ch_iters + pos_iter * ch_iters + ch_iter + 1
            print(f"[{current_iter}/{total_iters}] Running alpha={alpha_val}, pos={pos_iter}, ch={ch_iter}")

