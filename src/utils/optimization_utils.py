
from dataclasses import dataclass, field, fields
import numpy as np
import gurobipy as gp
from gurobipy import GRB



@dataclass
class ProblemParams:
    """
    Input parameters for the optimization problem.
    Initialization for G_comm, G_sens, and S_mat are required.
    """
    G_comm: np.ndarray = field(metadata={"info": "(N_ap, N_cu) normalized comm channel gains"})
    G_sens: np.ndarray = field(metadata={"info": "(N_ap, N_ap, N_tg) normalized sensing channel gains"})
    S_mat: np.ndarray = field(metadata={"info": "(N_ap, N_cu, N_cu) user-user correlation matrices"})

    alpha: float = field(default=0.5, metadata={"info": "Trade-off parameter b/w comm and sensing [default=0.5]"})
    rho_thresh: float = field(default=0.8, metadata={"info": "User spatial correlation threshold [default=0.8]"})
    interf_penalty: float = field(default=0.01, metadata={"info": "Penalty factor for interference [default=0.01]"})

    lambda_cu: np.ndarray = field(default=None, metadata={"info": "(N_cu,) comm user priority"})
    lambda_tg: np.ndarray = field(default=None, metadata={"info": "(N_tg,) target priority"})
    mu: np.ndarray = field(default=None, metadata={"info": "(N_ap,) AP TX mode priority [dependent: lambda_cu]"})
    C_rx: np.ndarray = field(default=None, metadata={"info": "(N_ap,) max number of targets an AP can listen to"})
    N_RF: np.ndarray = field(default=None, metadata={"info": "(N_ap,) number of RF chains per AP"})
    K_tx: np.ndarray = field(default=None, metadata={"info": "(N_tg,) max APs associated to a target for transmission"})
    K_rx: np.ndarray = field(default=None, metadata={"info": "(N_tg,) max APs associated to a target for reception"})

    modelMIPGap: float = field(default=1e-4, metadata={"info": "MIP gap for the solver [default=1e-4]"})
    modelTimeLimit: int = field(default=None, metadata={"info": "Time limit for the solver in seconds"})
    modelNodeLimit: int = field(default=None, metadata={"info": "Node limit for the solver"})
    modelThreads: int = field(default=None, metadata={"info": "Threads limit for the solver"})
    modelOutputFlag: bool = field(default=False, metadata={"info": "Solver output flag [default=False]"})

    N_ap: int = field(init=False, metadata={"info": "Number of  APs [internal]"})
    N_cu: int = field(init=False, metadata={"info": "Number of comm users [internal]"})
    N_tg: int = field(init=False, metadata={"info": "Number of targets [internal]"})

    def __post_init__(self):
        self._update_depencencies()

    def _update_depencencies(self):
        """
        Update dependent parameters based on the provided inputs.
        """
        self.N_ap = self.G_comm.shape[0]
        self.N_cu = self.G_comm.shape[1]
        self.N_tg = self.G_sens.shape[2]
        if self.lambda_cu is None:
            self.lambda_cu = np.ones(self.N_cu)
        elif np.isscalar(self.lambda_cu):
            self.lambda_cu = np.full(self.N_cu, self.lambda_cu)
        else:
            self.lambda_cu = np.asarray(self.lambda_cu)
            assert len(self.lambda_cu) == self.N_cu, "lambda_cu length must be equal to N_cu"
        if self.lambda_tg is None:
            self.lambda_tg = np.ones(self.N_tg)
        elif np.isscalar(self.lambda_tg):
            self.lambda_tg = np.full(self.N_tg, self.lambda_tg)
        else:
            self.lambda_tg = np.asarray(self.lambda_tg)
            assert len(self.lambda_tg) == self.N_tg, "lambda_tg length must be equal to N_tg"
        self.mu = np.sum(self.G_comm * self.lambda_cu.reshape(1, -1), axis=1)
        if self.C_rx is None:
            self.C_rx = np.full(self.N_ap, self.N_tg, dtype=int)
        elif np.isscalar(self.C_rx):
            self.C_rx = np.full(self.N_ap, self.C_rx, dtype=int)
        else:
            self.C_rx = np.asarray(self.C_rx, dtype=int)
            assert len(self.C_rx) == self.N_ap, "C_rx length must be equal to N_ap"
        if self.N_RF is None:
            self.N_RF = np.full(self.N_ap, 4, dtype=int)
        elif np.isscalar(self.N_RF):
            self.N_RF = np.full(self.N_ap, self.N_RF, dtype=int)
        else:
            self.N_RF = np.asarray(self.N_RF, dtype=int)
            assert len(self.N_RF) == self.N_ap, "N_RF length must be equal to N_ap"
        if self.K_tx is None:
            self.K_tx = np.full(self.N_tg, min(2, self.N_ap - 1), dtype=int)
        elif np.isscalar(self.K_tx):
            self.K_tx = np.full(self.N_tg, self.K_tx, dtype=int)
        else:
            self.K_tx = np.asarray(self.K_tx, dtype=int)
            assert len(self.K_tx) == self.N_tg, "K_tx length must be equal to N_tg"
        if self.K_rx is None:
            self.K_rx = np.full(self.N_tg, min(2, self.N_ap - 1), dtype=int)
        elif np.isscalar(self.K_rx):
            self.K_rx = np.full(self.N_tg, self.K_rx, dtype=int)
        else:
            self.K_rx = np.asarray(self.K_rx, dtype=int)
            assert len(self.K_rx) == self.N_tg, "K_rx length must be equal to N_tg"

    def change(self, update_dependencies: bool = True, **kwargs):
        """
        Change parameters and update dependent parameters if necessary.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Parameter '{key}' not found in ProblemParams.")
        if update_dependencies:
            self._update_depencencies()

    def summary(self, show_functions: bool = True):
        """
        Print a summary of the parameters and available methods.
        """
        import inspect
        width = 50
        print("=" * width)
        print("ProblemParams Reference".center(width))
        print("=" * width)
        title = " Parameters "
        print(title.center(width, "-"))
        for f in fields(self):
            info = f.metadata.get("info", "")
            print(f"{f.name:<15} : {info}")
        if show_functions:
            title = " Available Methods "
            print(title.center(width, "-"))
            for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
                if name.startswith("_"):  # skip internals
                    continue
                doc = method.__doc__.strip() if method.__doc__ else "No description"
                print(f"{name:<15} : {doc}")
        print("=" * width)



@dataclass
class Solution:
    tau: np.ndarray = field(metadata={"info": "(N_ap,) AP operation mode variable [1=Tx, 0=Rx]"})
    x: np.ndarray = field(metadata={"info": "(N_ap, N_cu) AP-user association variable [1=associated, 0=not]"})
    y_tx: np.ndarray = field(metadata={"info": "(N_ap, N_tg) TxAP-target association variable [1=associated, 0=not]"})
    y_rx: np.ndarray = field(metadata={"info": "(N_ap, N_tg) RxAP-target association variable [1=associated, 0=not]"})
    s: np.ndarray = field(metadata={"info": "(N_tg,) target scheduling variable [1=scheduled, 0=not]"})

    def summary(self):
        """
        Print a summary of the parameters and available methods.
        """
        width = 50
        print("=" * width)
        print("Solution Reference".center(width))
        print("=" * width)
        title = " Parameters "
        print(title.center(width, "-"))
        for f in fields(self):
            info = f.metadata.get("info", "")
            print(f"{f.name:<5} : {info}")
        print("=" * width)



# Optimization solver
def solve_problem(params: ProblemParams, print_status=True, return_status=False, return_objVal=False) -> Solution:
    """
    Solve the optimization problem using Gurobi solver.
    :param params: ProblemParams object containing the problem parameters
    :param print_status: Whether to print the optimization status
    :param return_status: Whether to return the optimization status
    :param return_objVal: Whether to return the objective value
    :return: Solution object containing the solution variables (and optionally status and objective value)
    """

    # ----- Define the optimization model
    model = gp.Model("ASSENT_model")

    # Decision Variables (binary)
    x = model.addVars(params.N_ap, params.N_cu, vtype=GRB.BINARY, name="x")
    y_tx = model.addVars(params.N_ap, params.N_tg, vtype=GRB.BINARY, name="y_tx")
    y_rx = model.addVars(params.N_tg, params.N_ap, vtype=GRB.BINARY, name="y_rx")
    tau = model.addVars(params.N_ap, vtype=GRB.BINARY, name="tau")
    s = model.addVars(params.N_tg, vtype=GRB.BINARY, name="s")

    # Auxiliary Variables
    z = model.addVars(params.N_ap, params.N_tg, params.N_ap, vtype=GRB.BINARY, name="z")
    w = model.addVars(params.N_ap, params.N_tg, params.N_ap, vtype=GRB.BINARY, name="w")
    v = model.addVars(params.N_ap, params.N_cu, params.N_cu, vtype=GRB.BINARY, name="v")

    # ----- Constraints

    # 1. AP Tx/Rx mode constraints
    for a in range(params.N_ap):
        for u in range(params.N_cu):
            model.addConstr(x[a, u] <= tau[a])
        for t in range(params.N_tg):
            model.addConstr(y_tx[a, t] <= tau[a])
            model.addConstr(y_rx[t, a] <= 1 - tau[a])

    # 2. RF chain constraints
    for a in range(params.N_ap):
        model.addConstr(
            gp.quicksum(x[a, u] for u in range(params.N_cu)) + gp.quicksum(y_tx[a, t] for t in range(params.N_tg)) <=
            params.N_RF[a])

    # 3. Target scheduling constraints
    for t in range(params.N_tg):
        model.addConstr(gp.quicksum(y_tx[a, t] for a in range(params.N_ap)) <= params.K_tx[t] * s[t])
        model.addConstr(gp.quicksum(y_rx[t, a] for a in range(params.N_ap)) >= s[t])
        model.addConstr(gp.quicksum(y_tx[a, t] for a in range(params.N_ap)) >= s[t])  # newly added to the problem
        model.addConstr(
            gp.quicksum(y_rx[t, a] for a in range(params.N_ap)) <= params.K_rx[t] * s[t])  # newly added to the problem

    # 4. User-User correlation constraints
    for a in range(params.N_ap):
        for u in range(params.N_cu):
            for up in range(params.N_cu):
                if u < up: model.addConstr(x[a, u] + x[a, up] <= 2 + params.rho_thresh - params.S_mat[a, u, up])

    # 5. Product linearization for z = s * y_tx * y_rx and w = y_tx * y_rx
    for a_t in range(params.N_ap):
        for t in range(params.N_tg):
            for a_r in range(params.N_ap):
                model.addConstr(w[a_t, t, a_r] <= y_tx[a_t, t])
                model.addConstr(w[a_t, t, a_r] <= y_rx[t, a_r])
                model.addConstr(w[a_t, t, a_r] >= y_tx[a_t, t] + y_rx[t, a_r] - 1)

                model.addConstr(z[a_t, t, a_r] <= s[t])
                model.addConstr(z[a_t, t, a_r] <= w[a_t, t, a_r])
                model.addConstr(z[a_t, t, a_r] >= s[t] + w[a_t, t, a_r] - 1)

    # 6. Target reception constraints
    for a in range(params.N_ap):
        model.addConstr(gp.quicksum(y_rx[t, a] for t in range(params.N_tg)) <= params.C_rx[a])

    # 7. Product linearization for v = x * x
    for a in range(params.N_ap):
        for u in range(params.N_cu):
            for up in range(params.N_cu):
                if u == up:
                    continue
                model.addConstr(v[a, u, up] <= x[a, u])
                model.addConstr(v[a, u, up] <= x[a, up])
                model.addConstr(v[a, u, up] >= x[a, u] + x[a, up] - 1)

    # ----- Objective Function

    # Formulation: interference-aware linear channel-based utility functions
    U_comm_ref = (np.sum(params.G_comm) - params.interf_penalty *
                  np.sum([params.S_mat[a, u, up] * params.G_comm[a, u] for a in range(params.N_ap) for u in
                          range(params.N_cu) for up in range(params.N_cu) if u < up]))
    U_sens_ref = np.sum(params.G_sens)

    comm_util = (gp.quicksum(
        params.lambda_cu[u] * params.G_comm[a, u] * x[a, u] for a in range(params.N_ap) for u in range(params.N_cu))
                 - params.interf_penalty * gp.quicksum(((params.lambda_cu[u] + params.lambda_cu[up]) / 2) * params.S_mat[a, u, up] * params.G_comm[a, u] * v[a, u, up]
                                                       for a in range(params.N_ap) for u in range(params.N_cu) for up in range(params.N_cu) if u < up)) / U_comm_ref
    sense_util = gp.quicksum(
        params.lambda_tg[t] * params.G_sens[a_t, a_r, t] * z[a_t, t, a_r] for a_t in range(params.N_ap) for a_r in
        range(params.N_ap) for t in range(params.N_tg)) / U_sens_ref
    tx_reward = gp.quicksum(params.mu[a] * tau[a] for a in range(params.N_ap))

    model.setObjective(params.alpha * comm_util + (1 - params.alpha) * sense_util + tx_reward, GRB.MAXIMIZE)

    # ----- Solve the optimization problem

    # Set solver parameters
    model.setParam("OutputFlag", params.modelOutputFlag)
    model.setParam("MIPGap", params.modelMIPGap)
    if params.modelTimeLimit is not None:
        model.setParam("TimeLimit", params.modelTimeLimit)
    if params.modelNodeLimit is not None:
        model.setParam("NodeLimit", params.modelNodeLimit)
    if params.modelThreads is not None:
        model.setParam("Threads", params.modelThreads)

    # Optimize the model
    model.optimize()

    if not params.modelOutputFlag and print_status:
        if model.status == gp.GRB.OPTIMAL:
            print("\n[OPT] --- Optimal solution found!")
        elif model.status == gp.GRB.INFEASIBLE:
            print("\n[OPT] --> Problem is infeasible!!")
        elif model.status == gp.GRB.TIME_LIMIT:
            print("\n[OPT] --> Stopped at time limit, feasible solution =", model.SolCount > 0)
        else:
            print("\n[OPT] --- Optimization ended with status", model.status)
        print(f"[OPT] --- Objective value: {model.objVal:.4f}, MIP gap: {model.MIPGap: .4f}, running time: {model.Runtime:.4f} sec")

    # ----- Extract Solution Values

    tau_values = np.array([tau[a].X for a in range(params.N_ap)])
    x_values = np.array([[x[a, u].X for u in range(params.N_cu)] for a in range(params.N_ap)])
    y_tx_values = np.array([[y_tx[a, t].X for t in range(params.N_tg)] for a in range(params.N_ap)])
    y_rx_values = np.array([[y_rx[t, a].X for t in range(params.N_tg)] for a in range(params.N_ap)])
    s_values = np.array([s[t].X for t in range(params.N_tg)])

    solution = Solution(tau=tau_values, x=x_values, y_tx=y_tx_values, y_rx=y_rx_values, s=s_values)
    if return_status and return_objVal:
        return solution, model.status, model.objVal
    elif return_status:
        return solution, model.status
    elif return_objVal:
        return solution, model.objVal
    else:
        return solution






def compute_milp_objective(input_params: dict, solution) -> dict:
    """
    Computes the objective value of the MILP problem.
    :param input_params: Dictionary {'G_comm', 'S_comm', 'G_sens', 'lambda_cu', 'lambda_tg', 'alpha', 'interf_penalty'}
    :param solution: Solution object or dictionary {'x', 'tau', 'y_tx', 'y_rx', 's'}
    :return: Dictionary with {'comm_util', 'sense_util', 'tx_reward', 'obj_val'}
    """
    from types import SimpleNamespace
    params = SimpleNamespace(**input_params)
    if isinstance(solution, dict):
        sol = SimpleNamespace(**solution)
    else:
        sol = solution

    A, U = params.G_comm.shape
    T = params.G_sens.shape[-1]
    x = np.asarray(sol.x, dtype=np.float64)         # [A,U]
    tau = np.asarray(sol.tau, dtype=np.float64)     # [A]
    ytx = np.asarray(sol.y_tx, dtype=np.float64)    # [A,T]
    yrx = np.asarray(sol.y_rx, dtype=np.float64)    # [A,T]
    s = np.asarray(sol.s, dtype=np.float64)         # [T]

    Gc = np.asarray(params.G_comm, dtype=np.float64)  # [A,U]
    Sc = np.asarray(params.S_comm, dtype=np.float64)  # [A,U,U]
    Gs = np.asarray(params.G_sens, dtype=np.float64)
    if np.isscalar(params.lambda_cu):
        params.lambda_cu = np.full(U, params.lambda_cu)
    lcu = np.asarray(params.lambda_cu, dtype=np.float64)  # [U]
    if np.isscalar(params.lambda_tg):
        params.lambda_tg = np.full(T, params.lambda_tg)
    ltg = np.asarray(params.lambda_tg, dtype=np.float64)  # [T]
    alpha = float(params.alpha)
    beta = float(params.interf_penalty)

    max_gain = max(np.max(Gc), np.max(Gs))
    Gcn = Gc / max_gain
    Gsn = Gs / max_gain

    mu = np.sum(Gcn * lcu.reshape(1, -1), axis=1)

    U_comm_ref = float(np.sum(Gcn) - beta * np.sum([Sc[a, u, up] * Gcn[a, u] for a in range(A)
                                                    for u in range(U) for up in range(U) if u < up]))
    U_sens_ref = np.sum(Gsn)

    # gain term
    comm_util_p1 = np.sum(lcu[None, :] * Gcn * x)
    # interference term: sum over u<up
    u, up = np.triu_indices(U, k=1)
    l_pair = 0.5 * (lcu[u] + lcu[up])  # [P], P=U*(U-1)/2
    # Stack per-AP contributions into [A, P] and sum axis=1 then overall:
    V = x[:, u] * x[:, up]  # [A, P]
    Sc_u = Sc[:, u, up]  # [A, P]
    Gcn_u = Gcn[:, u]  # [A, P]
    per_a = (l_pair[None, :] * Sc_u * Gcn_u * V).sum(axis=1)  # [A]
    comm_util_p2 = per_a.sum()
    comm_util = (comm_util_p1 - beta * comm_util_p2) / U_comm_ref

    sens_util = float(np.sum([ltg[t] * Gsn[a_t, a_r, t] * (ytx[a_t, t] * yrx[a_r, t] * s[t])
                              for a_t in range(A) for a_r in range(A) for t in range(T)])) / U_sens_ref

    tx_reward = np.sum(mu[a] * tau[a] for a in range(A))
    obj_val = alpha * comm_util + (1 - alpha) * sens_util + tx_reward
    obj_val_noreward = alpha * comm_util + (1 - alpha) * sens_util
    output = {'comm_util': comm_util, 'sens_util': sens_util, 'tx_reward': tx_reward,
              'obj_val': obj_val, 'obj_val_noreward': obj_val_noreward}
    return output




