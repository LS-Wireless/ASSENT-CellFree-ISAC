
from src.learning_v4.assent_data_utils_v4 import *
import src.utils.optimization_utils as opt

CONSOLE_RUN = True
nsamps = 200
results = data_loader(nsamps=nsamps, CONSOLE_RUN=CONSOLE_RUN)

objVal_gt = []
objVal_est = []
for i in range(nsamps):
    params = {'G_comm': results['G_comm'][i], 'S_comm': results['S_comm'][i], 'G_sens': results['G_sens'][i],
              'lambda_cu': results['lambda_cu'][i], 'lambda_tg': results['lambda_tg'][i],
              'alpha': results['alpha'][i], 'interf_penalty': results['interf_penalty']}
    solution = results['solution'][i]

    objVal_gt.append(results['opt_objVal'][i])
    milp_obj = opt.compute_milp_objective(params, solution, add_reward=True)
    objVal_est.append(milp_obj['obj_val'])
    print(milp_obj['obj_val'])

print(np.allclose(objVal_gt, objVal_est))
import matplotlib.pyplot as plt
plt.plot(objVal_est)
plt.show()

#%%

import os
import src.utils.library as lib

cwd = os.getcwd()
console_run = cwd + '/src/optimization/exp1_baseline' if CONSOLE_RUN else '../optimization/exp1_baseline'

exp1_run_id = 'run_03'
exp1_folder_path = os.path.join(console_run, exp1_run_id)

filename = f'2025-11-06-00-44_results.pkl'
file_path = os.path.join(exp1_folder_path, filename)
exp1_results = lib.load_results(file_path)

for i in range(len(exp1_results)):
    print(exp1_results[i]['opt_status'])
