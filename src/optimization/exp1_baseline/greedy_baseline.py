import numpy as np
import src.utils.optimization_utils as opt


def greedy_assign(
        G_comm: np.ndarray,  # (N_ap, N_ue)
        G_sens: np.ndarray,  # (N_ap, N_ap, N_tg) [tx, rx, t]
        K_tx: int = 2,
        K_rx: int = 2,
        N_rf=4,  # int or array-like (N_ap,)
        alpha: float | None = 0.6,  # controls TX-vs-RX mode bias; if None -> 0.5
        T_sched: int | None = None,  # if None, auto from capacity; else use e.g. s_milp.sum()
        max_aps_per_user: int = 1  # UE can attach to up to this many TX APs
):
    """
    Returns:
      tau  : (N_ap,)   AP mode {1=TX, 0=RX}
      x    : (N_ap,N_ue)  AP->UE association {0,1} (only from TX-mode APs)
      s    : (N_tg,)   target scheduled {0,1}
      y_tx : (N_ap,N_tg) AP acts as TX for target {0,1} (only τ=1)
      y_rx : (N_ap,N_tg) AP acts as RX for target {0,1} (only τ=0)
    """
    N_ap, N_ue = G_comm.shape
    assert G_sens.shape[0] == N_ap and G_sens.shape[1] == N_ap, "G_sens dims mismatch with N_ap"
    N_tg = G_sens.shape[2]

    # --- RF budgets ---
    if isinstance(N_rf, (int, np.integer)):
        rf_cap = np.full(N_ap, int(N_rf), dtype=int)
    else:
        rf_cap = np.asarray(N_rf, dtype=int).copy()
        assert rf_cap.shape == (N_ap,)

    # --- 1) Decide AP modes τ (TX vs RX) ---
    a = 0.5 if alpha is None else float(alpha)
    # Communication potential per AP (best UE this AP could serve)
    comm_pot = G_comm.max(axis=1) if N_ue > 0 else np.zeros(N_ap)

    # Sensing potentials per AP (disallow monostatic by zeroing diagonal)
    M = G_sens.copy()
    idx = np.arange(N_ap)
    M[idx, idx, :] = 0.0
    # As TX: best (rx,t) partner; As RX: best (tx,t) partner
    sens_tx_pot = M.max(axis=(1, 2)) if N_tg > 0 else np.zeros(N_ap)
    sens_rx_pot = M.max(axis=(0, 2)) if N_tg > 0 else np.zeros(N_ap)

    # α-weighted scores: prefer TX if comm is strong and/or TX-sensing potential is high
    score_tx = a * comm_pot + (1.0 - a) * sens_tx_pot
    score_rx = (1.0 - a) * sens_rx_pot

    tau = (score_tx >= score_rx).astype(int)  # 1=TX, 0=RX

    # Ensure we have at least one TX and one RX if sensing is present
    if N_tg > 0:
        if tau.sum() == 0:
            # flip the AP with the biggest (score_tx - score_rx) margin to TX
            flip = np.argmax(score_tx - score_rx)
            tau[flip] = 1
        if (N_ap - tau.sum()) == 0:
            flip = np.argmin(score_tx - score_rx)
            tau[flip] = 0

    # Initialize budgets split by mode
    tx_budget = rf_cap * tau  # budgets for TX-mode APs
    rx_budget = rf_cap * (1 - tau)  # budgets for RX-mode APs

    # --- 2) Greedy UE assignment on TX-mode APs ---
    x = np.zeros((N_ap, N_ue), dtype=int)
    # order users by best gain from any TX-mode AP
    if N_ue > 0 and tau.sum() > 0:
        ue_strength = (G_comm[tau == 1, :].max(axis=0) if np.any(tau == 1) else np.zeros(N_ue))
        ue_order = np.argsort(ue_strength)[::-1]
        for u in ue_order:
            # rank TX-mode APs for this user
            ap_rank = np.argsort(G_comm[:, u])[::-1]
            attached = 0
            for a_idx in ap_rank:
                if tau[a_idx] == 1 and tx_budget[a_idx] > 0:
                    x[a_idx, u] = 1
                    tx_budget[a_idx] -= 1
                    attached += 1
                    if attached == max_aps_per_user:
                        break

    # --- 3) Greedy target scheduling (TX from τ=1, RX from τ=0) ---
    s = np.zeros(N_tg, dtype=int)
    y_tx = np.zeros((N_ap, N_tg), dtype=int)
    y_rx = np.zeros((N_ap, N_tg), dtype=int)

    # available candidates by mode
    def tx_cand():
        return set(np.where((tau == 1) & (tx_budget > 0))[0].tolist())

    def rx_cand():
        return set(np.where((tau == 0) & (rx_budget > 0))[0].tolist())

    # Rough target scoring to decide order (upper bound using top-K across current budgets)
    def target_score(t):
        tx_ok = np.array(list(tx_cand()))
        rx_ok = np.array(list(rx_cand()))
        if tx_ok.size == 0 or rx_ok.size == 0:
            return 0.0
        Mt = M[:, :, t]  # diag already zeroed
        sub = Mt[np.ix_(tx_ok, rx_ok)]
        # optimistic bound: sum of row/col top-K (not exact but good for ranking)
        row_best = sub.max(axis=1) if sub.size else np.array([])
        col_best = sub.max(axis=0) if sub.size else np.array([])
        s_tx = np.sort(row_best)[-min(K_tx, row_best.size):].sum() if row_best.size else 0.0
        s_rx = np.sort(col_best)[-min(K_rx, col_best.size):].sum() if col_best.size else 0.0
        return float(s_tx + s_rx)

    scores = np.array([target_score(t) for t in range(N_tg)]) if N_tg > 0 else np.array([])
    tgt_order = np.argsort(scores)[::-1]

    # If T_sched not provided, cap by role capacities
    if N_tg > 0:
        cap_tx = int(tx_budget.sum()) // max(1, K_tx)
        cap_rx = int(rx_budget.sum()) // max(1, K_rx)
        cap_t = min(cap_tx, cap_rx)
        if T_sched is None:
            T_sched = int(min(N_tg, max(0, cap_t)))
        else:
            T_sched = int(min(T_sched, cap_t, N_tg))

    # Construct TX/RX sets for each selected target
    for t in tgt_order[: (T_sched or 0)]:
        if len(tx_cand()) < K_tx or len(rx_cand()) < K_rx:
            break  # not enough budgets
        Mt = M[:, :, t]  # (N_ap, N_ap) with diag=0
        T_set, R_set = set(), set()
        tx_ok, rx_ok = tx_cand(), rx_cand()

        # seed with best edge
        sub = Mt[np.ix_(list(tx_ok), list(rx_ok))]
        if sub.size == 0:
            continue
        i, j = np.unravel_index(np.argmax(sub, axis=None), sub.shape)
        a0 = list(tx_ok)[i]
        b0 = list(rx_ok)[j]
        T_set.add(a0)
        R_set.add(b0)

        # marginals
        def d_tx(a):
            if a in T_set or a not in tx_ok: return -np.inf
            return float(Mt[a, list(R_set)].sum()) if len(R_set) else float(Mt[a, list(rx_ok)].max())

        def d_rx(b):
            if b in R_set or b not in rx_ok: return -np.inf
            return float(Mt[list(T_set), b].sum()) if len(T_set) else float(Mt[list(tx_ok), b].max())

        while (len(T_set) < K_tx or len(R_set) < K_rx) and (len(tx_ok) > 0 and len(rx_ok) > 0):
            cand_tx = [(a, d_tx(a)) for a in tx_ok if a not in T_set]
            cand_rx = [(b, d_rx(b)) for b in rx_ok if b not in R_set]
            best_tx = max(cand_tx, key=lambda z: z[1]) if cand_tx else (None, -np.inf)
            best_rx = max(cand_rx, key=lambda z: z[1]) if cand_rx else (None, -np.inf)
            if best_tx[1] <= 0 and best_rx[1] <= 0:
                break
            if (best_tx[1] >= best_rx[1]) and best_tx[0] is not None and len(T_set) < K_tx:
                T_set.add(best_tx[0])
                tx_ok.discard(best_tx[0])
            elif best_rx[0] is not None and len(R_set) < K_rx:
                R_set.add(best_rx[0])
                rx_ok.discard(best_rx[0])
            else:
                break

        # finalize if feasible
        if len(T_set) < 1 or len(R_set) < 1:
            continue
        T_sel = list(T_set)[:K_tx]
        R_sel = list(R_set)[:K_rx]
        if min(tx_budget[T_sel].min(initial=1), rx_budget[R_sel].min(initial=1)) <= 0:
            continue

        for a in T_sel:
            y_tx[a, t] = 1
            tx_budget[a] -= 1
        for b in R_sel:
            y_rx[b, t] = 1
            rx_budget[b] -= 1
        s[t] = 1

    sol = opt.Solution(tau=tau.astype(int), x=x.astype(int), s=s.astype(int), y_tx=y_tx.astype(int),
                       y_rx=y_rx.astype(int))
    return sol



def greedy_assign_aligned_with_milp(G_comm, G_sens, N_rf, K_tx, K_rx,
                                    tau_milp, s_milp,
                                    max_aps_per_user=1):
    """
    Greedy baseline that *forces* AP modes (tau) and the target mask (s)
    to be exactly the MILP ones, preserving their column order.
    Then fills x, y_tx, y_rx greedily under budgets.
    """
    import numpy as np
    N_ap, N_ue = G_comm.shape
    N_tg = s_milp.size

    tau = tau_milp.astype(int).copy()
    s = s_milp.astype(int).copy()

    rf_cap = np.full(N_ap, int(N_rf), dtype=int) if np.isscalar(N_rf) else np.asarray(N_rf, int)
    tx_budget = rf_cap * tau
    rx_budget = rf_cap * (1 - tau)

    # --- Greedy UE assignment on TX-mode APs (order by best gain, but columns = UE index) ---
    x = np.zeros((N_ap, N_ue), dtype=int)
    if np.any(tau == 1) and N_ue > 0:
        ue_strength = (G_comm[tau == 1, :].max(axis=0) if np.any(tau == 1) else np.zeros(N_ue))
        ue_order = np.argsort(ue_strength)[::-1]
        for u in ue_order:
            ap_rank = np.argsort(G_comm[:, u])[::-1]
            attached = 0
            for a in ap_rank:
                if tau[a] == 1 and tx_budget[a] > 0:
                    x[a, u] = 1
                    tx_budget[a] -= 1
                    attached += 1
                    if attached == max_aps_per_user:
                        break

    # --- Greedy TX/RX for *exact* MILP-selected targets, in ascending target index ---
    y_tx = np.zeros((N_ap, N_tg), dtype=int)
    y_rx = np.zeros((N_ap, N_tg), dtype=int)

    selected_targets = np.where(s == 1)[0]  # exact MILP targets
    selected_targets = np.sort(selected_targets)  # enforce stable column order

    # forbid monostatic by construction: τ=1 are TX candidates, τ=0 are RX candidates
    M = G_sens.copy()
    np.fill_diagonal(M[:, :, 0], 0.0)  # harmless; we’ll never pair τ=1 and τ=0 same AP anyway

    for t in selected_targets:
        tx_cands = np.where((tau == 1) & (tx_budget > 0))[0]
        rx_cands = np.where((tau == 0) & (rx_budget > 0))[0]
        if tx_cands.size == 0 or rx_cands.size == 0:
            # no budget to realize this target; leave its column zero (still s[t]=1)
            continue

        Mt = M[:, :, t]
        sub = Mt[np.ix_(tx_cands, rx_cands)]
        if sub.size == 0:
            continue

        # pick TX set
        row_scores = sub.max(axis=1)  # best RX partner per TX
        tx_pick = tx_cands[np.argsort(row_scores)[-min(K_tx, row_scores.size):]]
        # pick RX set
        col_scores = sub.max(axis=0)  # best TX partner per RX
        rx_pick = rx_cands[np.argsort(col_scores)[-min(K_rx, col_scores.size):]]

        # mark and debit budgets
        for a in tx_pick:
            if tx_budget[a] > 0:
                y_tx[a, t] = 1;
                tx_budget[a] -= 1
        for b in rx_pick:
            if rx_budget[b] > 0:
                y_rx[b, t] = 1;
                rx_budget[b] -= 1

    sol = opt.Solution(tau=tau, x=x, s=s, y_tx=y_tx, y_rx=y_rx)
    return sol


import numpy as np


def greedy_assign_channel_only(
        G_comm: np.ndarray,  # (N_ap, N_ue)
        G_sens: np.ndarray,  # (N_ap, N_ap, N_tg)  [tx, rx, t]
        N_rf=4,  # int or array-like (N_ap,)
        K_tx: int = 2,
        K_rx: int = 2,
        max_aps_per_user: int = 1,
        T_sched: int | None = None,  # if None, schedule as many targets as RF budgets allow
        normalize_potentials: bool = True  # min-max normalize AP potentials before choosing tau
):
    """
    Greedy baseline driven purely by channel gains.

    Steps:
      1) Compute per-AP potentials from gains only:
         - comm_pot[a] = sum of top-L user gains for AP a (L = min(N_rf[a], N_ue))
         - sens_tx_pot[a] = max_{b,t} G_sens[a,b,t]   (as TX)
         - sens_rx_pot[a] = max_{b,t} G_sens[b,a,t]   (as RX)
         Optionally min-max normalize {comm_pot} and {sens_rx_pot} to avoid scale mismatch.
         Set tau[a] = 1 if comm_pot[a] >= sens_rx_pot[a], else 0 (ensure at least one TX and one RX).
      2) Assign users greedily to TX-mode APs by descending G_comm (≤ max_aps_per_user per UE, RF budgets respected).
      3) Rank targets by a simple score = best edge value max_{a in TX, b in RX} G_sens[a,b,t],
         then for each target pick TX and RX sets by row/col best under remaining budgets.

    Returns dict with int arrays: {'tau','x','s','y_tx','y_rx'}.
    """
    N_ap, N_ue = G_comm.shape
    assert G_sens.shape[0] == N_ap and G_sens.shape[1] == N_ap, "G_sens dims mismatch"
    N_tg = G_sens.shape[2]

    # --- RF capacity per AP
    if isinstance(N_rf, (int, np.integer)):
        rf_cap = np.full(N_ap, int(N_rf), dtype=int)
    else:
        rf_cap = np.asarray(N_rf, dtype=int).copy()
        assert rf_cap.shape == (N_ap,)

    # --- 1) Decide AP modes tau from gains only
    # Comm potential: sum of top-L user gains per AP (reflects RF capacity)
    L = np.minimum(rf_cap, N_ue)
    comm_pot = np.zeros(N_ap, dtype=float)
    if N_ue > 0:
        # sort descending per AP and sum top-L[a]
        sorted_comm = -np.sort(-G_comm, axis=1)  # (N_ap, N_ue)
        for a in range(N_ap):
            if L[a] > 0:
                comm_pot[a] = np.sum(sorted_comm[a, :L[a]])

    # Sensing potentials from G_sens
    if N_tg > 0:
        sens_tx_pot = G_sens.max(axis=(1, 2))  # as TX
        sens_rx_pot = G_sens.max(axis=(0, 2))  # as RX
    else:
        sens_tx_pot = np.zeros(N_ap, float)
        sens_rx_pot = np.zeros(N_ap, float)

    # Compare comm vs RX potentials to split roles (TX handles UEs; RX needed for echoes)
    c = comm_pot.copy()
    r = sens_rx_pot.copy()

    if normalize_potentials:
        # Min-max normalize each to [0,1] to mitigate scale mismatch between comm and sens gains
        def mm(x):
            xmin, xmax = float(np.min(x)), float(np.max(x))
            return (x - xmin) / (xmax - xmin + 1e-12) if xmax > xmin else np.zeros_like(x)

        c = mm(c);
        r = mm(r)

    tau = (c >= r).astype(int)  # 1 = TX, 0 = RX

    # Ensure at least one TX and one RX if sensing exists
    if N_tg > 0:
        if tau.sum() == 0:
            tau[np.argmax(c - r)] = 1
        if (N_ap - tau.sum()) == 0:
            tau[np.argmin(c - r)] = 0

    # Budgets by mode
    tx_budget = rf_cap * tau
    rx_budget = rf_cap * (1 - tau)

    # --- 2) UE associations (only from TX-mode APs)
    x = np.zeros((N_ap, N_ue), dtype=int)
    if N_ue > 0 and np.any(tau == 1):
        # order UEs by best available TX-AP gain
        ue_strength = G_comm[tau == 1, :].max(axis=0) if np.any(tau == 1) else np.zeros(N_ue)
        ue_order = np.argsort(ue_strength)[::-1]
        for u in ue_order:
            # rank all APs by gain; pick those with tau=1 and budget
            ap_rank = np.argsort(G_comm[:, u])[::-1]
            attached = 0
            for a in ap_rank:
                if tau[a] == 1 and tx_budget[a] > 0:
                    x[a, u] = 1
                    tx_budget[a] -= 1
                    attached += 1
                    if attached == max_aps_per_user:
                        break

    # --- 3) Target scheduling by gain only
    s = np.zeros(N_tg, dtype=int)
    y_tx = np.zeros((N_ap, N_tg), dtype=int)
    y_rx = np.zeros((N_ap, N_tg), dtype=int)

    if N_tg > 0 and np.any(tau == 1) and np.any(tau == 0):
        # Simple target score: best edge value with current candidate pools
        def score_t(t):
            tx_ok = np.where((tau == 1) & (tx_budget > 0))[0]
            rx_ok = np.where((tau == 0) & (rx_budget > 0))[0]
            if tx_ok.size == 0 or rx_ok.size == 0: return 0.0
            sub = G_sens[np.ix_(tx_ok, rx_ok, [t])][:, :, 0]
            return float(sub.max()) if sub.size else 0.0

        scores = np.array([score_t(t) for t in range(N_tg)], dtype=float)
        tgt_order = np.argsort(scores)[::-1]

        # Capacity-based default T_sched
        if T_sched is None:
            cap_tx = int(tx_budget.sum()) // max(1, K_tx)
            cap_rx = int(rx_budget.sum()) // max(1, K_rx)
            T_sched = int(min(N_tg, max(0, min(cap_tx, cap_rx))))

        # Greedy per-target TX/RX selection using row/col best with remaining budgets
        for t in tgt_order[:T_sched]:
            tx_ok = np.where((tau == 1) & (tx_budget > 0))[0]
            rx_ok = np.where((tau == 0) & (rx_budget > 0))[0]
            if tx_ok.size == 0 or rx_ok.size == 0:
                continue

            Mt = G_sens[:, :, t]

            # pick TX: top K_tx by their best RX partner
            if tx_ok.size:
                row_best = Mt[np.ix_(tx_ok, rx_ok)].max(axis=1) if rx_ok.size else np.zeros(tx_ok.size)
                tx_pick = tx_ok[np.argsort(row_best)[-min(K_tx, tx_ok.size):]]
            else:
                tx_pick = np.array([], dtype=int)

            # pick RX: top K_rx by their best TX partner
            if rx_ok.size:
                col_best = Mt[np.ix_(tx_ok, rx_ok)].max(axis=0) if tx_ok.size else np.zeros(rx_ok.size)
                rx_pick = rx_ok[np.argsort(col_best)[-min(K_rx, rx_ok.size):]]
            else:
                rx_pick = np.array([], dtype=int)

            if tx_pick.size == 0 or rx_pick.size == 0:
                continue

            # debit budgets & mark
            feas = np.all(tx_budget[tx_pick] > 0) and np.all(rx_budget[rx_pick] > 0)
            if not feas:
                continue

            for a in tx_pick:
                y_tx[a, t] = 1;
                tx_budget[a] -= 1
            for b in rx_pick:
                y_rx[b, t] = 1;
                rx_budget[b] -= 1
            s[t] = 1

    # --- Canonicalize (keep columns aligned with global indices; zero-out unscheduled)
    y_tx *= s[np.newaxis, :]
    y_rx *= s[np.newaxis, :]
    sol = opt.Solution(tau=tau, x=x, s=s, y_tx=y_tx, y_rx=y_rx)
    return sol



def greedy_assign_comm_only(G_comm, N_rf=4, N_tg=0, max_aps_per_user=1):
    N_ap, N_ue = G_comm.shape
    rf_cap = np.full(N_ap, int(N_rf), dtype=int) if np.isscalar(N_rf) else np.asarray(N_rf, int)
    # all TX, no RX role used
    tau = np.ones(N_ap, dtype=int)
    tx_budget = rf_cap.copy()

    x = np.zeros((N_ap, N_ue), dtype=int)
    if N_ue > 0:
        ue_strength = G_comm.max(axis=0)
        for u in np.argsort(ue_strength)[::-1]:
            ap_rank = np.argsort(G_comm[:, u])[::-1]
            attached = 0
            for a in ap_rank:
                if tx_budget[a] > 0:
                    x[a, u] = 1
                    tx_budget[a] -= 1
                    attached += 1
                    if attached == max_aps_per_user:
                        break

    # no sensing
    s = np.zeros(N_tg, dtype=int)
    # safer: let caller pass N_tg; here we derive zero-sized arrays:
    y_tx = np.zeros((N_ap, N_tg), dtype=int)
    y_rx = np.zeros((N_ap, N_tg), dtype=int)
    sol = opt.Solution(tau=tau, x=x, s=s, y_tx=y_tx, y_rx=y_rx)
    return sol




def greedy_assign_sens_only(
    G_sens: np.ndarray,          # (N_ap, N_ap, N_tg) [tx, rx, t]
    N_rf=4,
    K_tx: int = 2,
    K_rx: int = 2,
    N_ue: int = 0,               # for evaluator shape; x is zeros (N_ap, N_ue)
    T_sched: int | None = None,
    normalize_potentials: bool = False
) -> opt.Solution:
    N_ap, _, N_tg = G_sens.shape

    rf_cap = (np.full(N_ap, int(N_rf), dtype=int)
              if isinstance(N_rf, (int, np.integer))
              else np.asarray(N_rf, dtype=int).copy())
    assert rf_cap.shape == (N_ap,)

    # Mode split by sensing potential (no monostatic)
    sens_tx_pot = G_sens.max(axis=(1, 2))  # as TX
    sens_rx_pot = G_sens.max(axis=(0, 2))  # as RX
    c_tx, c_rx = sens_tx_pot.astype(float), sens_rx_pot.astype(float)
    if normalize_potentials:
        def mm(x):
            xmin, xmax = float(np.min(x)), float(np.max(x))
            return (x - xmin) / (xmax - xmin + 1e-12) if xmax > xmin else np.zeros_like(x)
        c_tx, c_rx = mm(c_tx), mm(c_rx)

    tau = (c_tx >= c_rx).astype(int)
    if N_tg > 0:
        if tau.sum() == 0: tau[np.argmax(c_tx - c_rx)] = 1
        if tau.sum() == N_ap: tau[np.argmin(c_tx - c_rx)] = 0

    tx_budget = (rf_cap * tau).astype(int)
    rx_budget = (rf_cap * (1 - tau)).astype(int)

    x    = np.zeros((N_ap, N_ue), dtype=int)
    s    = np.zeros(N_tg, dtype=int)
    y_tx = np.zeros((N_ap, N_tg), dtype=int)
    y_rx = np.zeros((N_ap, N_tg), dtype=int)

    if N_tg == 0 or (tau.sum() == 0) or (tau.sum() == N_ap):
        sol = opt.Solution(tau=tau, x=x, s=s, y_tx=y_tx, y_rx=y_rx)
        return sol

    # ---- scoring with broadcasted slice (no np.ix_) ----
    def score_t(t: int) -> float:
        tx_ok = np.where((tau == 1) & (tx_budget > 0))[0]
        rx_ok = np.where((tau == 0) & (rx_budget > 0))[0]
        if tx_ok.size == 0 or rx_ok.size == 0:
            return 0.0
        sub = G_sens[tx_ok[:, None], rx_ok[None, :], t]      # (n_tx, n_rx)
        row_best = sub.max(axis=1) if sub.size else np.array([])
        col_best = sub.max(axis=0) if sub.size else np.array([])
        s_tx = np.sort(row_best)[-min(K_tx, row_best.size):].sum() if row_best.size else 0.0
        s_rx = np.sort(col_best)[-min(K_rx, col_best.size):].sum() if col_best.size else 0.0
        return float(s_tx + s_rx)

    scores = np.array([score_t(t) for t in range(N_tg)], dtype=float)
    tgt_order = np.argsort(scores)[::-1]

    if T_sched is None:
        cap_tx = int(tx_budget.sum()) // max(1, K_tx)
        cap_rx = int(rx_budget.sum()) // max(1, K_rx)
        T_sched = int(min(N_tg, max(0, min(cap_tx, cap_rx))))

    for t in tgt_order[:T_sched]:
        tx_ok = np.where((tau == 1) & (tx_budget > 0))[0]
        rx_ok = np.where((tau == 0) & (rx_budget > 0))[0]
        if tx_ok.size == 0 or rx_ok.size == 0:
            continue

        sub = G_sens[tx_ok[:, None], rx_ok[None, :], t]      # (n_tx, n_rx)
        if sub.size == 0:
            continue

        row_best = sub.max(axis=1)
        col_best = sub.max(axis=0)
        tx_pick = tx_ok[np.argsort(row_best)[-min(K_tx, tx_ok.size):]]
        rx_pick = rx_ok[np.argsort(col_best)[-min(K_rx, rx_ok.size):]]

        if tx_pick.size == 0 or rx_pick.size == 0:
            continue
        if np.any(tx_budget[tx_pick] <= 0) or np.any(rx_budget[rx_pick] <= 0):
            continue

        for a in tx_pick: y_tx[a, t] = 1; tx_budget[a] -= 1
        for b in rx_pick: y_rx[b, t] = 1; rx_budget[b] -= 1
        s[t] = 1

    y_tx *= s[np.newaxis, :]
    y_rx *= s[np.newaxis, :]

    sol = opt.Solution(tau=tau, x=x, s=s, y_tx=y_tx, y_rx=y_rx)
    return sol








