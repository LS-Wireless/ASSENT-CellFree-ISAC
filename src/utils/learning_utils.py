
import numpy as np
import itertools
import torch
from torch_geometric.data import HeteroData


def prepare_snapshot_data(coordinates, G_comm, S_comm, G_sens, alpha, lambda_cu, lambda_tg, solution, N_RF, M_a):
    """
    Prepares one snapshot of the simulation data for creating a PyG graph.
    :param coordinates: All coordinates (ap_positions, user_positions, target_positions) for one snapshot.
    :param G_comm: (N_ap x N_cu) channel gains matrix.
    :param S_comm: (N_ap x N_cu x N_cu) user-user correlation matrix.
    :param G_sens: (N_ap x N_ap x N_tg) sensing channel gains matrix.
    :param alpha: Alpha value for that snapshot.
    :param lambda_cu: (N_cu,) array of user priority weights lambda_cu.
    :param lambda_tg: (N_tg,) array of target priority weights lambda_tg.
    :param solution: Optimization solution object.
    :param N_RF: (Scalar) Number of RF chains per AP.
    :param M_a: (Scalar) Number of antennas per AP.
    :return: A snapshot dictionary containing all data for one sample.
    """
    ap_positions = coordinates['ap_positions']
    user_positions = coordinates['user_positions']
    target_positions = coordinates['target_positions']

    num_aps = ap_positions.shape[0]
    num_users = user_positions.shape[0]
    num_targets = target_positions.shape[0]
    N_antennas_per_ap = np.full((num_aps, 1), M_a)
    N_rf_per_ap = np.full((num_aps, 1), N_RF)

    # --- 2. Process Node Features ---
    # We stack coordinates with the policy parameters (lambdas, alpha)
    # [x_coord, y_coord, lambda, alpha]
    snapshot_data = {'ap_features': np.hstack(
        [ap_positions, N_antennas_per_ap, N_rf_per_ap, np.full((num_aps, 1), alpha)]),
                     'user_features': np.hstack(
                         [user_positions, lambda_cu.reshape(-1, 1), np.full((num_users, 1), alpha)]),
                     'target_features': np.hstack(
                         [target_positions, lambda_tg.reshape(-1, 1), np.full((num_targets, 1), alpha)])}
    # --- Calculate the per-link average correlation ---
    avg_corr_per_link = np.zeros_like(G_comm)
    for a in range(G_comm.shape[0]):
        for u in range(num_users):
            # Calculate the mean correlation for user u with all other users, from AP a's perspective
            other_users_mask = np.ones(num_users, dtype=bool)
            other_users_mask[u] = False
            if num_users > 1:
                avg_corr_per_link[a, u] = np.mean(S_comm[a, u, other_users_mask])
            else:
                avg_corr_per_link[a, u] = 0

    # --- 3. Process AP-User Edges (with ENRICHED features) ---
    ap_user_src, ap_user_dst = np.meshgrid(np.arange(num_aps), np.arange(num_users))
    snapshot_data['ap_user_edges'] = np.vstack([ap_user_src.flatten(), ap_user_dst.flatten()])

    # Flatten the matrices to match the edge order
    G_comm_flat = G_comm.T.flatten().reshape(-1, 1)
    avg_corr_flat = avg_corr_per_link.T.flatten().reshape(-1, 1)

    # Stack the features horizontally to create a 2D edge feature matrix
    snapshot_data['ap_user_gains'] = np.hstack([G_comm_flat, avg_corr_flat])
    snapshot_data['x_solution'] = solution.x.T.flatten()

    # --- 4. Process User-User Edges (Interference) ---
    # S_comm is (N_ap, N_cu, N_cu). For the graph, we need a single N_cu x N_cu
    # A good approach is to average over the AP dimension.
    s_comm_avg = np.mean(S_comm, axis=0)

    # # Create edges for all unique user pairs (u, up) where u != up
    # user_pairs = np.array(list(itertools.combinations(range(num_users), 2))).T
    # # We add the reverse edges to make the graph undirected
    # snapshot_data['user_user_edges'] = np.hstack([user_pairs, user_pairs[[1, 0], :]])
    # # Get the correlation values for these pairs
    # corr_values = s_comm_avg[user_pairs[0], user_pairs[1]]
    # snapshot_data['user_user_corr'] = np.hstack([corr_values, corr_values]).reshape(-1, 1)

    # --- 5. Process AP-Target Edges (Sensing) ---
    ap_target_src, ap_target_dst = np.meshgrid(np.arange(num_aps), np.arange(num_targets))
    snapshot_data['ap_target_edges'] = np.vstack([ap_target_src.flatten(), ap_target_dst.flatten()])

    # --- Use the monostatic gain G_sens[a,a,t] as the edge feature ---
    # np.diagonal extracts the elements where the first two axes are equal.
    # This results in a matrix of shape (N_tg, N_ap).
    monostatic_gains = np.diagonal(G_sens, axis1=0, axis2=1)

    # We need to flatten this in the same order as our edge_index.
    # The meshgrid order is (ap0,t0), (ap1,t0), (ap2,t0)...
    # So we must transpose the monostatic_gains matrix before flattening.
    snapshot_data['ap_target_merit'] = monostatic_gains.T.flatten().reshape(-1, 1)

    # Get the sensing association labels (ytx, yrx)
    snapshot_data['ytx_solution'] = solution.y_tx.T.flatten()
    snapshot_data['yrx_solution'] = solution.y_rx.T.flatten()

    # --- 6. Add Node-Level Labels ---
    snapshot_data['tau_solution'] = solution.tau
    snapshot_data['s_solution'] = solution.s

    return snapshot_data



def create_graph_sample(snapshot_data):
    """
    Creates a PyG graph object for one sample.
    :param snapshot_data: One snapshot dictionary containing all data for one sample.
    :return: HeteroData: A PyG graph object for one sample.
    """
    data = HeteroData()

    # --- 1. Define Node Features ---
    # Node features should include location, policy params, etc.
    # [x_coord, y_coord, lambda, alpha]
    data['ap'].x = torch.tensor(snapshot_data['ap_features'], dtype=torch.float)
    data['user'].x = torch.tensor(snapshot_data['user_features'], dtype=torch.float)
    data['target'].x = torch.tensor(snapshot_data['target_features'], dtype=torch.float)

    # --- 2. Define Edges and Edge Features ---
    # Edge index is a [2, num_edges] tensor representing (source, destination) pairs.
    # Communication Links (AP <-> User)
    data['ap', 'serves', 'user'].edge_index = torch.tensor(snapshot_data['ap_user_edges'], dtype=torch.long)
    data['ap', 'serves', 'user'].edge_attr = torch.tensor(snapshot_data['ap_user_gains'], dtype=torch.float)

    # User Interference Links (User <-> User)
    # data['user', 'interferes', 'user'].edge_index = torch.tensor(snapshot_data['user_user_edges'], dtype=torch.long)
    # data['user', 'interferes', 'user'].edge_attr = torch.tensor(snapshot_data['user_user_corr'], dtype=torch.float)

    # Sensing Links (AP <-> Target)
    data['ap', 'senses', 'target'].edge_index = torch.tensor(snapshot_data['ap_target_edges'], dtype=torch.long)
    # Edge attribute could be the one-way path quality
    data['ap', 'senses', 'target'].edge_attr = torch.tensor(snapshot_data['ap_target_merit'], dtype=torch.float)

    # --- 3. Store the Labels (MILP Solution) ---
    # We store the ground-truth labels directly on the graph objects.
    # Node-level labels
    data['ap'].y_tau = torch.tensor(snapshot_data['tau_solution'], dtype=torch.float)
    data['target'].y_s = torch.tensor(snapshot_data['s_solution'], dtype=torch.float)

    # Edge-level labels
    data['ap', 'serves', 'user'].y_x = torch.tensor(snapshot_data['x_solution'], dtype=torch.float)
    data['ap', 'senses', 'target'].y_ytx = torch.tensor(snapshot_data['ytx_solution'], dtype=torch.float)
    data['ap', 'senses', 'target'].y_yrx = torch.tensor(snapshot_data['yrx_solution'], dtype=torch.float)

    return data


