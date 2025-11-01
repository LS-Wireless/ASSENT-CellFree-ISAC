
from src.learning_v4.assent_data_utils_v4 import *
import src.utils.optimization_utils as opt



results = data_loader(nsamps=100)

objVal_gt = []
objVal_est = []
for i in range(100):
    params = {'G_comm': results['G_comm'][i], 'S_comm': results['S_comm'][i], 'G_sens': results['G_sens'][i],
              'lambda_cu': results['lambda_cu'][i], 'lambda_tg': results['lambda_tg'][i],
              'alpha': results['alpha'][i], 'interf_penalty': results['interf_penalty']}
    solution = results['solution'][i]

    objVal_gt.append(results['opt_objVal'][i])
    milp_obj = opt.compute_milp_objective(params, solution)
    objVal_est.append(milp_obj['obj_val'])
    print(milp_obj['sens_util'])

print(np.allclose(objVal_gt, objVal_est))


