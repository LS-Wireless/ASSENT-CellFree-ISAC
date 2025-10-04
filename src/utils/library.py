
import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform
import pandas as pd


# ------------------------------------------
# Functions for generating network topology
# ------------------------------------------

# >> Function to place nodes with minimum distance to other nodes
def placement_with_min_dist(N, env_x, env_y, min_dist, current_positions=None, min_dist_to_current_positions=None, max_attempts=5000,
                            failed_action='warning', random_state=None):
    """
    Place N nodes in a 2D environment of size env_x by env_y with specified minimum distances.
    :param N: The number of nodes
    :param env_x: The environment width (x-axis)
    :param env_y: The environment height (y-axis)
    :param min_dist: The minimum distance between nodes
    :param current_positions: Previously placed nodes
    :param min_dist_to_current_positions: The minimum distance to previously placed nodes
    :param max_attempts: The maximum number of attempts to place nodes
    :param failed_action: If 'warning', issue a warning if placement fails; if 'error', raise an error if placement fails
    :param random_state: The random state
    :return: The positions of placed nodes
    """
    rng = np.random.default_rng(random_state)
    if current_positions is None:
        current_positions = np.empty((0, 2))
    if min_dist_to_current_positions is None:
        min_dist_to_current_positions = 0
    new_positions = np.empty((0, 2))
    if max(min_dist, min_dist_to_current_positions) > env_x or max(min_dist, min_dist_to_current_positions) > env_y:
        raise ValueError("min_dist and/or min_dist_to_current_positions exceeds env_x and/or env_y dimensions.")
    env_max_dim = max(env_x, env_y)
    attempts = 0
    while len(new_positions) < N:
        x = rng.uniform(0, env_x)
        y = rng.uniform(0, env_y)
        new_point = np.column_stack([x, y])
        if len(new_positions) == 0 and len(current_positions) == 0:
            dists = env_max_dim
            dists_to_current_positions = env_max_dim
        elif len(new_positions) == 0 and len(current_positions) > 0:
            dists = env_max_dim
            dists_to_current_positions = np.linalg.norm(current_positions - new_point, axis=1)
        elif len(new_positions) > 0 and len(current_positions) == 0:
            dists = np.linalg.norm(new_positions - new_point, axis=1)
            dists_to_current_positions = env_max_dim
        else:
            dists = np.linalg.norm(new_positions - new_point, axis=1)
            dists_to_current_positions = np.linalg.norm(current_positions - new_point, axis=1)
        if np.any(dists < min_dist) or np.any(dists_to_current_positions < min_dist_to_current_positions):
            attempts += 1
            if attempts > max_attempts:
                if failed_action == 'error':
                    raise RuntimeError("Could not place nodes with given min_dist and/or min_dist_to_current_positions. Try reducing them.")
                else:
                    warnings.warn("Could not satisfy min_dist and/or min_dist_to_current_positions among all nodes.")
                    new_positions = np.vstack([new_positions, new_point])
                    continue
            continue
        new_positions = np.vstack([new_positions, new_point])
    return new_positions



# >> Function to generate AP positions
def generate_AP_positions(N_ap, env_x, env_y, mode='uniform', min_dist=None, circle_radius=None, line_point1=None, line_point2=None,
                          random_state=None):
    """
    Generate AP positions in a 2D environment of size env_x by env_y.
    :param N_ap: The number of APs
    :param env_x: The environment width (x-axis)
    :param env_y: The environment height (y-axis)
    :param mode: The mode of placement ('uniform', 'circle', or 'line')
    :param min_dist: The minimum distance between APs
    :param circle_radius: If mode='circle', radius of circle
    :param line_point1: If mode='line', starting point of line
    :param line_point2: If mode='line', ending point of line
    :param random_state: The random state
    :return: AP positions
    """
    rng = np.random.default_rng(random_state)
    if mode == 'uniform':
        if min_dist is None:
            x_coords = rng.uniform(0, env_x, N_ap)
            y_coords = rng.uniform(0, env_y, N_ap)
            ap_positions = np.column_stack([x_coords, y_coords])
        else:
            ap_positions = placement_with_min_dist(N=N_ap, env_x=env_x, env_y=env_y, min_dist=min_dist, current_positions=None,
                                                   failed_action='warning', random_state=random_state)
    elif mode == 'circle':
        if circle_radius is None:
            raise ValueError("circle_radius must be specified for 'circle' mode.")
        if circle_radius > env_x or circle_radius > env_y:
            warnings.warn("circle_radius exceeds env_x and/or env_y dimensions.")
        x_coords = circle_radius * np.cos(np.linspace(0, 2 * np.pi, N_ap, endpoint=False)) + env_x / 2
        y_coords = circle_radius * np.sin(np.linspace(0, 2 * np.pi, N_ap, endpoint=False)) + env_y / 2
        ap_positions = np.column_stack([x_coords, y_coords])
    elif mode == 'line':
        if line_point1 is None or line_point2 is None:
            raise ValueError("line_point1 and line_point2 must be specified for 'line' mode.")
        if line_point1[0] == line_point2[0] and line_point1[1] == line_point2[1]:
            raise ValueError("line_point1 and line_point2 must be different points.")
        x_coords = np.linspace(line_point1[0], line_point2[0], N_ap)
        y_coords = np.linspace(line_point1[1], line_point2[1], N_ap)
        ap_positions = np.column_stack([x_coords, y_coords])
    else:
        raise ValueError("Invalid mode. Choose 'uniform', 'circle', or 'line'.")
    return ap_positions




# >> Function to generate entity (user/target) positions
def generate_entity_positions(N_ent, env_x, env_y, mode='uniform', min_entity_dist=None, min_entity2current_dist=None, current_positions=None,
                              circle_radius=None, line_point1=None, line_point2=None, random_state=None):
    """
    Generate entity (user/target) positions in a 2D environment of size env_x by env_y.
    :param N_ent: The number of entities
    :param env_x: The enviroment width (x-axis)
    :param env_y: The enviroment height (y-axis)
    :param mode: The mode of placement ('uniform', 'circle', or 'line')
    :param min_entity_dist: The minimal distance between entities
    :param min_entity2current_dist: The minimal distance between entities and current positions
    :param current_positions: Positions of previously placed nodes
    :param circle_radius: If mode='circle', radius of circle
    :param line_point1: If mode='line', starting point of line
    :param line_point2: If mode='line', ending point of line
    :param random_state: The random state
    :return: The entity positions
    """
    rng = np.random.default_rng(random_state)
    if mode == 'uniform':
        if min_entity_dist is None and min_entity2current_dist is None:
            x_coords = rng.uniform(0, env_x, N_ent)
            y_coords = rng.uniform(0, env_y, N_ent)
            entity_positions = np.column_stack([x_coords, y_coords])
        elif min_entity_dist is not None and min_entity2current_dist is None:
            entity_positions = placement_with_min_dist(N=N_ent, env_x=env_x, env_y=env_y, min_dist=min_entity_dist, current_positions=None,
                                                       failed_action='warning', random_state=random_state)
        elif min_entity_dist is None and min_entity2current_dist is not None:
            if current_positions is None:
                raise ValueError("current_positions must be provided when min_entity2current_dist is specified.")
            entity_positions = placement_with_min_dist(N=N_ent, env_x=env_x, env_y=env_y, min_dist=0, current_positions=current_positions,
                                                       min_dist_to_current_positions=min_entity2current_dist, failed_action='warning', random_state=random_state)
        else:
            if current_positions is None:
                raise ValueError("current_positions must be provided when min_entity2current_dist is specified.")
            entity_positions = placement_with_min_dist(N=N_ent, env_x=env_x, env_y=env_y, min_dist=min_entity_dist, current_positions=current_positions,
                                                       min_dist_to_current_positions=min_entity2current_dist, failed_action='warning', random_state=random_state)
    elif mode == 'circle':
        if circle_radius is None:
            raise ValueError("circle_radius must be specified for 'circle' mode.")
        if circle_radius > env_x or circle_radius > env_y:
            warnings.warn("circle_radius exceeds env_x and/or env_y dimensions.")
        x_coords = circle_radius * np.cos(np.linspace(0, 2 * np.pi, N_ent, endpoint=False)) + env_x / 2
        y_coords = circle_radius * np.sin(np.linspace(0, 2 * np.pi, N_ent, endpoint=False)) + env_y / 2
        entity_positions = np.column_stack([x_coords, y_coords])
    elif mode == 'line':
        if line_point1 is None or line_point2 is None:
            raise ValueError("line_point1 and line_point2 must be specified for 'line' mode.")
        if line_point1[0] == line_point2[0] and line_point1[1] == line_point2[1]:
            raise ValueError("line_point1 and line_point2 must be different points.")
        x_coords = np.linspace(line_point1[0], line_point2[0], N_ent)
        y_coords = np.linspace(line_point1[1], line_point2[1], N_ent)
        entity_positions = np.column_stack([x_coords, y_coords])
    else:
        raise ValueError("Invalid mode. Choose 'uniform', 'circle', or 'line'.")
    return entity_positions



def create_geometry_dataframe(ap_positions, user_positions, target_positions, h_ap=10, h_cu=1.5, h_t=1.5):
    """
    Creates a DataFrame containing the geometry information (distances and angles) between APs, communication users, and targets.
    :param ap_positions: AP positions as a (N_ap x 2) array
    :param user_positions: User positions as a (N_cu x 2) array
    :param target_positions: Target positions as a (N_t x 2) array
    :param h_ap: Height of APs
    :param h_cu: Height of communication users
    :param h_t: Height of targets
    :return: Geometry DataFrame
    """
    geometry_df = pd.DataFrame(columns=['AP_id', 'AP_CU_d2D', 'AP_CU_d3D', 'AP_CU_azimuth_deg', 'AP_CU_elevation_deg',
                                        'AP_T_d2D', 'AP_T_d3D', 'AP_T_azimuth_deg', 'AP_T_elevation_deg']).reset_index(drop=True)
    for ap_id in range(ap_positions.shape[0]):
        # Distances and angles to communication users
        ap_cu_d2D = np.linalg.norm(ap_positions[ap_id, :] - user_positions, axis=1)
        ap_cu_d3D = np.sqrt(ap_cu_d2D ** 2 + (h_ap - h_cu) ** 2)
        ap_cu_azimuth_deg = np.rad2deg(np.arctan2(user_positions[:, 1] - ap_positions[ap_id, 1], user_positions[:, 0] - ap_positions[ap_id, 0]))
        ap_cu_azimuth_deg = (ap_cu_azimuth_deg + 360) % 360
        ap_cu_elevation_deg = np.rad2deg(np.arctan2(ap_cu_d2D, h_ap - h_cu))
        geometry_df.loc[ap_id, 'AP_id'] = ap_id
        geometry_df.loc[ap_id, 'AP_CU_d2D'] = ap_cu_d2D
        geometry_df.loc[ap_id, 'AP_CU_d3D'] = ap_cu_d3D
        geometry_df.loc[ap_id, 'AP_CU_azimuth_deg'] = ap_cu_azimuth_deg
        geometry_df.loc[ap_id, 'AP_CU_elevation_deg'] = ap_cu_elevation_deg
        # Distances and angles to targets
        ap_t_d2D = np.linalg.norm(ap_positions[ap_id, :] - target_positions, axis=1)
        ap_t_d3D = np.sqrt(ap_t_d2D ** 2 + (h_ap - h_t) ** 2)
        ap_t_azimuth_deg = np.rad2deg(np.arctan2(target_positions[:, 1] - ap_positions[ap_id, 1], target_positions[:, 0] - ap_positions[ap_id, 0]))
        ap_t_azimuth_deg = (ap_t_azimuth_deg + 360) % 360
        ap_t_elevation_deg = np.rad2deg(np.arctan2(ap_t_d2D, h_ap - h_t))
        geometry_df.loc[ap_id, 'AP_T_d2D'] = ap_t_d2D
        geometry_df.loc[ap_id, 'AP_T_d3D'] = ap_t_d3D
        geometry_df.loc[ap_id, 'AP_T_azimuth_deg'] = ap_t_azimuth_deg
        geometry_df.loc[ap_id, 'AP_T_elevation_deg'] = ap_t_elevation_deg
    return geometry_df




# ------------------------------------------------
# Functions for generating communication channels
# ------------------------------------------------

# UCA array response function
def uca_array_response(M_a: int, uca_radius: float, wavelength: float, azimuth_deg: float, elevation_deg=90) -> np.ndarray:
    """
    UCA array response function.
    :param M_a: The number of antennas in the array
    :param uca_radius: The UCA radius
    :param wavelength: The carrier wavelength
    :param azimuth_deg: The azimuth AoA/AoD in degrees
    :param elevation_deg: The elevation AoA/AoD in degrees
    :return: The array response
    """
    k = 2 * np.pi / wavelength
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    phi_n = 2 * np.pi * np.arange(M_a) / M_a
    response = np.exp(1j * k * uca_radius * np.sin(elevation_rad) * np.cos(azimuth_rad - phi_n))
    return response


def get_umi_los_probability(distance_2D: float) -> float:
    """
    Calculates the LOS probability for the UMi-Street Canyon scenario.
    Based on 3GPP TR 38.901, Table 7.4.2-1.
    :param distance_2D: The 2D distance between AP and user in meters.
    :return (float): The probability of having a Line-of-Sight link.
    """
    if distance_2D <= 18:
        return 1.0
    else:
        prob_los = (18 / distance_2D) + (1 - (18/distance_2D)) * np.exp(-distance_2D / 36)
        return prob_los


def generate_3gpp_umi_parameters(is_los: bool, carrier_freq_GHz: float, ap_cu_d2D, h_ap=10.0, h_cu=1.5, h_E=1.0, c=3.0e8, ch_random_state=None) -> dict:
    """
    Generates the 3GPP UMi-Street Canyon scenario parameters.
    :param is_los: True if the link is LOS, False if it is NLOS
    :param carrier_freq_GHz: Carrier frequency in GHz
    :param ap_cu_d2D: Distance between AP and communication user in 2D
    :param h_ap: Height of AP (default for UMi: 10 meters)
    :param h_cu: Height of communication user (default for UMi: 1.5 meters)
    :param h_E: Effective environment height (default for UMi: 1.0 meters)
    :param c: Speed of light in meters per second (default for UMi: 3.0e8 m/s)
    :param ch_random_state: Random state for channel generation
    :return: Dictionary of parameters ('.sf_db', '.k_factor_db', '.pl_db', '.asd_rad', '.zsd_rad')
    """
    ch_rng = np.random.default_rng(ch_random_state)
    K_factor_mu_dB = 9.0
    K_factor_sigma_dB = 5.0
    params = {}
    # Calculate breakpoint distance for UMi
    d_BP_prime = 4 * (h_ap - h_E) * (h_cu - h_E) * carrier_freq_GHz * 1e9 / c
    ap_cu_d3D = np.sqrt(ap_cu_d2D**2 + (h_ap - h_cu)**2)
    PL_1 = 32.4 + 21 * np.log10(ap_cu_d3D) + 20 * np.log10(carrier_freq_GHz)
    PL_2 = 32.4 + 40 * np.log10(ap_cu_d3D) + 20 * np.log10(carrier_freq_GHz) - 9.5 * np.log10(d_BP_prime ** 2 + (h_ap - h_cu) ** 2)
    PL_los = PL_1 if ap_cu_d2D <= d_BP_prime else PL_2
    if is_los:  # LOS
        params['sf_db'] = 4.0
        params['k_factor_db'] = ch_rng.normal(K_factor_mu_dB, K_factor_sigma_dB)
        params['pl_db'] = PL_los

        # ASD (Azimuth Spread of Departure)
        mu_log_asd = -0.05 * np.log10(1 + carrier_freq_GHz) + 1.21
        sigma_log_asd = 0.41
        log_asd_deg = ch_rng.normal(mu_log_asd, sigma_log_asd)
        params['asd_rad'] = np.deg2rad(10**log_asd_deg)

        # ZSD (Zenith Spread of Departure)
        mu_1 = -14.8 * (ap_cu_d2D / 1000) + 0.01 * np.abs(h_ap - h_cu) + 0.83
        mu_log_zsd = max(-0.21, mu_1)
        sigma_log_zsd = 0.35
        log_zsd_deg = ch_rng.normal(mu_log_zsd, sigma_log_zsd)
        params['zsd_rad'] = np.deg2rad(10**log_zsd_deg)
    else:   # NLOS
        params['sf_db'] = 7.82
        params['k_factor_db'] = -100.0  # Effectively Rayleigh

        # NLOS pathloss
        PL_prime = 35.3 * np.log10(ap_cu_d3D) + 22.4 + 21.3 * np.log10(carrier_freq_GHz) - 0.3 * (h_cu - 1.5)
        params['pl_db'] = max(PL_los, PL_prime)

        # ASD (Azimuth Spread of Departure)
        mu_log_asd = -0.23 * np.log10(1 + carrier_freq_GHz) + 1.53
        sigma_log_asd = 0.11 * np.log10(1 + carrier_freq_GHz) + 0.33
        log_asd_deg = ch_rng.normal(mu_log_asd, sigma_log_asd)
        params['asd_rad'] = np.deg2rad(10**log_asd_deg)

        # ZSD (Zenith Spread of Departure)
        mu_1 = -3.1 * (ap_cu_d2D / 1000) + 0.01 * max(h_cu - h_ap, 0) + 0.2
        mu_log_zsd = max(-0.5, mu_1)
        sigma_log_zsd = 0.35
        log_zsd_deg = ch_rng.normal(mu_log_zsd, sigma_log_zsd)
        params['zsd_rad'] = np.deg2rad(10**log_zsd_deg)

    return params



def generate_correlated_shadowing(user_positions: np.ndarray, sf_std_devs: np.ndarray, decorr_distance: float = 50.0, ch_random_state=None) -> np.ndarray:
    """
    Generates correlated shadowing values for all users with respect to a single AP.
    :param user_positions: A (N_cu x 2) array of user (x, y) coordinates.
    :param sf_std_devs: A 1D array of size N_cu containing the shadowing STD (in dB) for each user, which depends on their LOS/NLOS condition.
    :param decorr_distance: The decorrelation distance (D_corr) in meters.
    :param ch_random_state: The random state to use for channel generation.
    :return np.ndarray: A 1D array of size N_cu with the correlated shadowing values (in dB).
    """
    ch_rng = np.random.default_rng(ch_random_state)
    num_users = user_positions.shape[0]

    # Step 1: Calculate the pairwise distances between all users.
    # pdist computes the condensed distance matrix, squareform converts it to a full KxK matrix.
    inter_user_distances = squareform(pdist(user_positions))

    # Step 2: Calculate the correlation coefficient matrix based on the exponential model.
    # This matrix rho_kj has 1s on the diagonal.
    correlation_coeffs = np.exp(-inter_user_distances / decorr_distance)

    # Step 3: Build the full covariance matrix.
    # This correctly handles cases where different users have different shadowing STDs.
    # The (k, j)-th element will be sigma_k * sigma_j * rho_kj.
    sigma_vector = sf_std_devs.reshape(-1, 1)  # Ensure it's a column vector
    outer_prod_of_stds = sigma_vector @ sigma_vector.T
    covariance_matrix = correlation_coeffs * outer_prod_of_stds

    # Add a small epsilon to the diagonal for numerical stability before Cholesky.
    epsilon = 1e-6
    covariance_matrix += np.eye(num_users) * epsilon

    # Step 4: Perform Cholesky decomposition. L is a lower-triangular matrix.
    # We want to find L such that L @ L.T = covariance_matrix.
    try:
        L = np.linalg.cholesky(covariance_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Shadowing covariance matrix was not positive-definite. Using nearest SPD matrix.")
        # Fallback for rare numerical issues
        from scipy.linalg import eigh
        eigval, eigvec = eigh(covariance_matrix)
        eigval[eigval < 0] = epsilon
        covariance_matrix = eigvec @ np.diag(eigval) @ eigvec.T
        L = np.linalg.cholesky(covariance_matrix)

    # Step 5: Generate independent standard normal random variables.
    w = ch_rng.normal(0, 1, (num_users, 1))

    # Step 6: Create the correlated shadowing values by applying the "filter" L.
    # The resulting vector 's' will have the desired covariance structure.
    correlated_shadowing_values = (L @ w).flatten()

    return correlated_shadowing_values



def create_spatial_correlation_matrix(M_a: int, uca_radius: float, carrier_wavelength: float, mean_aod: float, angular_spread_asd: float, num_samples: int = 500) -> np.ndarray:
    """
    Creates the spatial correlation matrix R_au for a UCA using numerical integration of a Truncated Laplacian Power Angular Spectrum (PAS) in the azimuth plane.
    :param M_a: The number of antennas in the UCA.
    :param uca_radius: The radius of the UCA in meters.
    :param carrier_wavelength: The wavelength of the carrier in meters.
    :param mean_aod: The mean AOD of the UCA (the azimuth angle of the user) in radians.
    :param angular_spread_asd: The AOD spread (ASD) based on 3GPP in radians.
    :param num_samples: The number of samples to use for numerical integration.
    :return np.ndarray: The spatial correlation matrix R_au.
    """
    # Step 1: Define the grid of angles for numerical integration
    thetas = np.linspace(-np.pi, np.pi, num_samples)

    # Step 2: Evaluate the Truncated Laplacian PAS on the grid
    angle_diff = thetas - mean_aod
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    pas_values = np.exp(-np.sqrt(2) * np.abs(angle_diff) / angular_spread_asd)

    # Step 3: Normalize the PAS
    pas_values /= np.sum(pas_values)

    # Step 4: Pre-calculate UCA antenna positions
    phi_antennas = 2 * np.pi * np.arange(M_a) / M_a

    # Step 5: Build the matrix R_au using numerical summation
    R_au = np.zeros((M_a, M_a), dtype=complex)
    wavenumber = 2 * np.pi / carrier_wavelength
    for p in range(M_a):
        for q in range(M_a):
            phase_diff_exponent = wavenumber * uca_radius * (np.cos(phi_antennas[p] - thetas) - np.cos(phi_antennas[q] - thetas))
            complex_exponentials = np.exp(1j * phase_diff_exponent)
            integral_sum = np.sum(pas_values * complex_exponentials)
            R_au[p, q] = integral_sum

    # Step 6: Final trace normalization
    current_trace = np.trace(R_au)
    if np.abs(current_trace) > 1e-9:
        R_au = (M_a / current_trace) * R_au

    return R_au


def generate_nlos_component(R_au: np.ndarray, ch_random_state=None) -> np.ndarray:
    """
    Generates the normalized, spatially correlated NLOS channel vector.
    :param R_au: The M_a x M_a spatial correlation matrix.
    :param ch_random_state: The random state to use for channel generation.
    :return np.ndarray: The M_a x 1 normalized NLOS channel vector.
    """
    ch_rng = np.random.default_rng(ch_random_state)
    M_a = R_au.shape[0]

    # Add a small identity matrix for numerical stability before Cholesky decomposition
    epsilon = 1e-9
    try:
        # L is the "filter" such that L @ L.T.conj() = R_mk
        L = np.linalg.cholesky(R_au + np.eye(M_a) * epsilon)
    except np.linalg.LinAlgError:
        # Fallback for rare cases where matrix is not positive definite
        print("Warning: NLOS correlation matrix R_au was not positive-definite. Returning uncorrelated channel vector.")
        # In a real sim, you might regenerate parameters or use a fallback
        # For now, we return a simple uncorrelated channel
        L = np.eye(M_a)

    # Generate i.i.d. complex Gaussian noise vector z_mk (mean 0, variance 1)
    z_au = (ch_rng.normal(0, 1, (M_a, 1)) + 1j * ch_rng.normal(0,1, (M_a, 1))) / np.sqrt(2)

    # Apply the correlation "filter" L to the noise vector
    h_nlos_normalized = L @ z_au

    return h_nlos_normalized


def generate_commLink_params(ap_cu_d2D, h_ap=10, h_cu=1.5, fc=3.0e9, c=3e8, ch_random_state=None):
    """
    Generates the parameters for a single communication link (one AP to all users).
    :param ap_cu_d2D: 2D distance vector from AP to all users.
    :param h_ap: Height of the AP in meters (UMi defaults to 10m)
    :param h_cu: Height of the communication user in meters (UMi defaults to 1.5m)
    :param fc: Carrier frequency in Hz (default: 3 GHz)
    :param c: Speed of light in m/s
    :param ch_random_state: Random state to use for channel generation.
    :return: DataFrame with the link parameters ('user_id', 'is_los', 3gpp_params_dict).
    """
    ch_rng = np.random.default_rng(ch_random_state)
    N_cu = ap_cu_d2D.shape[0]
    carrier_freq_GHz = fc / 1e9
    link_params_list = []
    for u in range(N_cu):
        ap_cu_d2D_u = ap_cu_d2D[u]
        prob_los_u = get_umi_los_probability(ap_cu_d2D_u)
        is_los = ch_rng.random() < prob_los_u
        umi_3gpp_params_u = generate_3gpp_umi_parameters(is_los=is_los, carrier_freq_GHz=carrier_freq_GHz, ap_cu_d2D=ap_cu_d2D_u, h_ap=h_ap, h_cu=h_cu,
                                                         c=c, ch_random_state=ch_random_state)
        link_params_full = {'user_id': u, 'is_los': is_los, **umi_3gpp_params_u}
        link_params_list.append(link_params_full)
    link_params_df = pd.DataFrame(link_params_list)
    return link_params_df


def generate_commChannel_singleAP(M_a, ap_position, user_positions, uca_radius=0.05, h_ap=10, h_cu=1.5, fc=3.0e9, c=3e8, mode='3gpp',
                                  return_df=False, ch_random_state=None):
    """
    Generates channels from a single AP to all users.
    :param M_a: The number of antennas in the UCA
    :param ap_position: The position of the specific AP
    :param user_positions: The positions of all users
    :param uca_radius: The radius of the UCA in meters (default: 0.05m)
    :param h_ap: The height of the AP in meters (UMi defaults to 10m)
    :param h_cu: The height of the communication user in meters (UMi defaults to 1.5m)
    :param fc: The carrier frequency in Hz (default: 3 GHz)
    :param c: The speed of light in m/s
    :param mode: Channel generation mode, currently only '3gpp' is supported
    :param return_df: If True, return a DataFrame with link parameters and channel vectors. If False, return only the channel matrix H_a
    :param ch_random_state: The random state to use for channel generation
    :return: Matrix H_a with size (N_cu x M_a) and (optionally) DataFrame with link parameters
    """
    N_cu = user_positions.shape[0]
    carrier_wavelength = c / fc
    # Distances and angles
    ap_cu_d2D = np.linalg.norm(ap_position - user_positions, axis=1)
    ap_cu_azimuth_deg = np.rad2deg(np.arctan2(user_positions[:, 1] - ap_position[1], user_positions[:, 0] - ap_position[0]))
    ap_cu_azimuth_deg = (ap_cu_azimuth_deg + 360) % 360
    ap_cu_azimuth_rad = np.deg2rad(ap_cu_azimuth_deg)
    ap_cu_elevation_deg = np.rad2deg(np.arctan2(ap_cu_d2D, h_ap - h_cu))
    # Get link parameters for all users
    link_params_df = generate_commLink_params(ap_cu_d2D, h_ap=h_ap, h_cu=h_cu, fc=fc, c=c, ch_random_state=ch_random_state)
    sf_std_db = link_params_df['sf_db'].values
    correlated_sf_vals_db = generate_correlated_shadowing(user_positions, sf_std_devs=sf_std_db, decorr_distance=50.0, ch_random_state=ch_random_state)
    link_params_df['pl_total_db'] = link_params_df['pl_db'] + correlated_sf_vals_db

    link_params_df['h_au'] = None
    H_a = np.zeros((N_cu, M_a), dtype=complex)
    for u in range(N_cu):
        azimuth_u = ap_cu_azimuth_rad[u]
        azimuth_u_deg = ap_cu_azimuth_deg[u]
        asd_u = link_params_df['asd_rad'].values[u]
        elevation_u_deg = ap_cu_elevation_deg[u]
        h_los = uca_array_response(M_a=M_a, uca_radius=uca_radius, wavelength=carrier_wavelength, azimuth_deg=azimuth_u_deg, elevation_deg=elevation_u_deg)
        pl_total_u = link_params_df['pl_total_db'].values[u]
        pl_u_linear = 10 ** (-(pl_total_u / 10))
        R_au = create_spatial_correlation_matrix(M_a=M_a, uca_radius=uca_radius, carrier_wavelength=carrier_wavelength, mean_aod=azimuth_u,
                                                 angular_spread_asd=asd_u, num_samples=500)
        h_nlos = generate_nlos_component(R_au, ch_random_state=ch_random_state)
        h_nlos = np.squeeze(h_nlos)
        if link_params_df['is_los'].values[u]:
            k_factor_linear = 10 ** (link_params_df['k_factor_db'].values[u] / 10)
            h_small_scale = np.sqrt(k_factor_linear / (k_factor_linear + 1)) * h_los + np.sqrt(1 / (k_factor_linear + 1)) * h_nlos
        else:
            h_small_scale = h_los
        h_au = np.sqrt(pl_u_linear) * h_small_scale
        link_params_df.at[u, 'h_au'] = h_au
        H_a[u, :] = h_au

    if return_df:
        return H_a, link_params_df
    else:
        return H_a



def generate_commChannels(M_a, ap_positions, user_positions, uca_radius=0.05, h_ap=10, h_cu=1.5, fc=3.0e9, c=3e8, mode='3gpp', ch_random_state=None):
    """
    Generates all communication channels for a given set of APs and users.
    :param M_a: Number of antennas
    :param ap_positions: AP positions (N_ap x 2)
    :param user_positions: User positions (N_cu x 2)
    :param uca_radius: UCA radius in meters (default: 0.05m)
    :param h_ap: Height of the AP in meters (default: 10m)
    :param h_cu: Height of the communication user in meters (default: 1.5m)
    :param fc: Carrier frequency in Hz (default: 3 GHz)
    :param c: Speed of light in m/s (default: 300 m/s)
    :param mode: Mode of channel generation (default: '3gpp')
    :param ch_random_state: Channel random state (default: None)
    :return: H_mat: Matrix of channel vectors (N_ap x N_cu x M_a)
    """
    N_ap = ap_positions.shape[0]
    M_a = np.atleast_1d(M_a)
    M_a_vec = np.resize(M_a, N_ap)

    H_mat = np.zeros((N_ap, user_positions.shape[0], M_a.max()), dtype=complex)
    for ap_id in range(N_ap):
        H_a = generate_commChannel_singleAP(M_a=M_a_vec[ap_id], ap_position=ap_positions[ap_id,:], user_positions=user_positions, uca_radius=uca_radius,
                                            h_ap=h_ap, h_cu=h_cu, fc=fc, c=c, mode=mode, return_df=False, ch_random_state=ch_random_state)
        H_mat[ap_id, :, :] = H_a

    return H_mat




def extract_commLink_features(H_mat, normalize=False):
    """
    Generates channel gains and spatial correlations for each AP to all users.
    :param H_mat: Total channel matrix (N_ap x N_cu x M_a)
    :param normalize: Whether to normalize the user channel vectors (default: False)
    :return: G_mat: Channel gains (N_ap x N_cu) and S_mat: Spatial correlations (N_ap x N_cu x N_cu)
    """

    H_norm = H_mat / np.linalg.norm(H_mat, axis=-1, keepdims=True)
    H_mat = H_norm if normalize else H_mat
    G_mat = np.linalg.norm(H_mat, axis=-1) ** 2
    S_mat = np.abs(H_norm.conj() @ H_norm.transpose(0, 2, 1))

    return G_mat, S_mat



# -------------------------------------------
# Functions for generating Sensing channels
# -------------------------------------------


def get_target_rcs(target_class='car', class_mean_rcs=None, target_velocity=None, v_ref=1.0, random_state=None):
    """
    Generates a target RCS based on its class and velocity.
    :param target_class: Target class ('car', 'human', or 'drone')
    :param class_mean_rcs: The mean RCS for the target class in m^2. If None, a random mean will be chosen.
    :param target_velocity: The target velocity in m/s. If None, a default velocity will be assumed based on the class.
    :param v_ref: Reference velocity sensitivity for the aspect angle variation (default: 1.0 m/s)
    :param random_state: Random state for reproducibility
    :return: RCS value in m^2
    """
    rng = np.random.default_rng(random_state)
    if target_class == 'car':
        if class_mean_rcs is None:
            class_mean_db = rng.uniform(low=10, high=20)
        else:
            class_mean_db = 10 * np.log10(class_mean_rcs)
        target_velocity = target_velocity if target_velocity is not None else rng.uniform(low=0, high=10)
        sigma_aspect_min = 1.0
        sigma_aspect_max = 3.0
        v_max = 30.0
    elif target_class == 'human':
        if class_mean_rcs is None:
            class_mean_db = rng.uniform(low=-10, high=0)
        else:
            class_mean_db = 10 * np.log10(class_mean_rcs)
        target_velocity = target_velocity if target_velocity is not None else rng.uniform(low=0, high=2)
        sigma_aspect_min = 1.0
        sigma_aspect_max = 3.0
        v_max = 3.0
    elif target_class == 'drone':
        if class_mean_rcs is None:
            class_mean_db = rng.uniform(low=-20, high=-5)
        else:
            class_mean_db = 10 * np.log10(class_mean_rcs)
        target_velocity = target_velocity if target_velocity is not None else rng.uniform(low=0, high=10)
        sigma_aspect_min = 0.1
        sigma_aspect_max = 2.0
        v_max = 15.0
    else:
        raise ValueError("Invalid target class. Choose from 'car', 'human', or 'drone'.")

    sigma_aspect = sigma_aspect_min + (sigma_aspect_max - sigma_aspect_min) * (np.log10(1 + (target_velocity / v_ref)) / np.log10(1 + v_max / v_ref))
    delta_aspect_db = rng.normal(loc=0, scale=sigma_aspect)
    rcs_db = class_mean_db + delta_aspect_db
    rcs = 10 ** (rcs_db / 10)
    return rcs



def get_target_los_prob(ap_target_d3D, mode='3gpp', target_velocity=None, prev_measurement=None):
    """
    Computes the probability of LOS for a target based on distance and other factors.
    :param ap_target_d3D: AP-target 3D distance in meters
    :param mode: LOS probability model ('3gpp' or 'logistic')
    :param target_velocity: Target velocity in m/s (used in 'logistic' mode)
    :param prev_measurement: Previous SNR measurement in dB (used in 'logistic' mode)
    :return: LOS probability
    """
    if mode == '3gpp':
        prob_los = get_umi_los_probability(ap_target_d3D)
    elif mode == 'logistic':
        w_baseline = 1.0
        w_dist = -0.5
        w_vel = 0.1
        w_prev_measurement = 0.5
        snr_threshold_db = 10
        target_velocity = target_velocity if target_velocity is not None else 0
        prev_measurement_term = 0 if prev_measurement is None else w_prev_measurement * (prev_measurement - snr_threshold_db)
        z = w_baseline + (w_dist * np.log10(ap_target_d3D)) + (w_vel * target_velocity) + prev_measurement_term
        prob_los = 1 / (1 + np.exp(-z))
    else:
        raise ValueError("Invalid mode. Choose either '3gpp' or 'logistic'.")

    return prob_los



def extract_sensLink_features(ap_positions, target_positions, relative_velocities, h_ap=10, h_tg=1.5, target_classes=None, target_mean_rcs=None,
                              prob_los_mode='3gpp', prev_measurements=None, random_state=None):
    """
    Generates sensing link parameters for all AP-Target pairs.
    :param ap_positions: AP positions (N_ap x 2)
    :param target_positions: Target positions (N_tg x 2)
    :param relative_velocities: Relative velocities of targets w.r.t. each AP (N_ap x N_tg)
    :param h_ap: Height of the APs in meters
    :param h_tg: Height of the targets in meters
    :param target_classes: Target classes (list of size N_tg with 'car', 'human', or 'drone'). If None, all targets are assumed to be 'car'.
    :param target_mean_rcs: Target mean RCS values (array of size N_tg). If None, random means will be chosen based on the class.
    :param prob_los_mode: Target LOS probability model ('3gpp' or 'logistic')
    :param prev_measurements: Previous SNR measurements for targets (array of size N_tg). Used only if prob_los_mode is 'logistic'.
    :param random_state: Random state for reproducibility
    :return: DataFrame with sensing link features ('ap_id', 'target_id', 'target_distance', 'target_los_prob', 'target_rcs_estimate')
    """
    N_ap = ap_positions.shape[0]
    N_tg = target_positions.shape[0]
    all_records = []
    for a in range(N_ap):
        ap_position = ap_positions[a,:]
        target_velocities = relative_velocities[a, :]
        # Distances
        ap_tg_d2D = np.linalg.norm(ap_position - target_positions, axis=1)
        ap_tg_d3D = np.sqrt(ap_tg_d2D ** 2 + (h_ap - h_tg) ** 2)

        for t in range(N_tg):
            prev_measurement = prev_measurements[t] if prev_measurements is not None else None
            target_velocity = target_velocities[t]
            los_prob = get_target_los_prob(ap_tg_d3D[t], mode=prob_los_mode, target_velocity=target_velocity, prev_measurement=prev_measurement)
            tg_class = target_classes[t] if target_classes is not None else 'car'
            tg_mean_rcs = target_mean_rcs[t] if target_mean_rcs is not None else None
            rcs = get_target_rcs(target_class=tg_class, class_mean_rcs=tg_mean_rcs, target_velocity=target_velocity, random_state=random_state)
            all_records.append({'ap_id': a, 'target_id': t, 'target_distance': ap_tg_d3D[t], 'target_los_prob': los_prob,
                                'target_rcs_estimate': rcs})

    records_df = pd.DataFrame(all_records)
    return records_df




def get_sensChannel_gains(sens_channel_df, wavelength=3e8/3e9, dB_scale=False, normalize_losWeight=True, w_los=1.0, w_nlos=0.2):
    """
    Computes the sensing channel gains for all TxAP-Target-RxAP pairs.
    :param sens_channel_df: DataFrame with sensing link parameters ('ap_id', 'target_id', 'target_distance', 'target_los_prob', 'target_rcs_estimate')
    :param wavelength: Wavelength in meters (default: 0.1m for 3 GHz)
    :param dB_scale: Whether to return gains in dB scale (default: False)
    :param normalize_losWeight: Whether to normalize the LOS probability weights (default: True)
    :param w_los: Value for LOS probability weight (default: 1.0)
    :param w_nlos: Value for NLOS probability weight (default: 0.2)
    :return: G_tg_mat: 3D array of sensing channel gains tx-rx-target: (N_ap x N_ap x N_tg)
    """
    N_ap = len(np.unique(sens_channel_df['ap_id']))
    N_tg = len(np.unique(sens_channel_df['target_id']))
    w_max = w_nlos + (w_los - w_nlos) * (np.max(sens_channel_df['target_los_prob']))

    G_tg_mat = np.zeros((N_ap, N_ap, N_tg))
    for atx in range(N_ap):
        for arx in range(N_ap):
            for tg in range(N_tg):
                # atx to tg
                d_atxtg = sens_channel_df.loc[(sens_channel_df['ap_id'] == atx) & (sens_channel_df['target_id'] == tg), 'target_distance'].values[0]
                rcs_atxtg = sens_channel_df.loc[(sens_channel_df['ap_id'] == atx) & (sens_channel_df['target_id'] == tg), 'target_rcs_estimate'].values[0]
                plos_atxtg = sens_channel_df.loc[(sens_channel_df['ap_id'] == atx) & (sens_channel_df['target_id'] == tg), 'target_los_prob'].values[0]
                w_atxtg = w_nlos + (w_los - w_nlos) * plos_atxtg
                w_atxtg = w_atxtg / w_max if normalize_losWeight else w_atxtg

                # tg to arx
                d_tgarx = sens_channel_df.loc[(sens_channel_df['ap_id'] == arx) & (sens_channel_df['target_id'] == tg), 'target_distance'].values[0]
                rcs_tgarx = sens_channel_df.loc[(sens_channel_df['ap_id'] == arx) & (sens_channel_df['target_id'] == tg), 'target_rcs_estimate'].values[0]
                plos_tgarx = sens_channel_df.loc[(sens_channel_df['ap_id'] == arx) & (sens_channel_df['target_id'] == tg), 'target_los_prob'].values[0]
                w_tgarx = w_nlos + (w_los - w_nlos) * plos_tgarx
                w_tgarx = w_tgarx / w_max if normalize_losWeight else w_tgarx

                g_txtgrx = (1 / (d_atxtg ** 2)) * (1 / (d_tgarx ** 2)) * ((rcs_atxtg + rcs_tgarx) / 2) * w_atxtg * w_tgarx
                G_tg_mat[atx, arx, tg] = g_txtgrx
    # Free-space pathloss factor
    G_tg_mat = G_tg_mat * ((wavelength ** 2) / ((4 * np.pi) ** 3))
    if dB_scale:
        G_tg_mat_db = 10 * np.log10(G_tg_mat)
        return G_tg_mat_db
    else:
        return G_tg_mat





# ------------------------------
# Functions for saving results
# ------------------------------

from dataclasses import asdict
import json
import os

def save_dataclass_hybrid(params, save_path, filename='params', save_json=True, save_arrays=True, print_log=True):
    """
    Saves non-np.ndarray parameters of a dataclass as JSON and np.ndarray parameters separately as NPZ.
    :param params: Dataclass object
    :param save_path: Save path
    :param filename: Filename
    :param save_json: Whether to save JSON (default: True)
    :param save_arrays: Whether to save arrays (default: True)
    :param print_log: Whether to show print statements (default: True)
    :return: No return value
    """
    # Split arrays and scalars
    params_dict = asdict(params)
    arrays_dict = {}
    for key, val in params_dict.items():
        if isinstance(val, np.ndarray):
            arrays_dict[key] = val
            params_dict[key] = f'np.ndarray with shape: {val.shape}'

    # Save scalars as JSON
    if save_json:
        file_path = os.path.join(save_path, f'{filename}.json')
        with open(file_path, "w") as f:
            json.dump(params_dict, f, indent=2)
        if print_log:
            title = " Saved Files "
            print(title.center(50, "-"))
            cwd = os.getcwd()
            rel_path = os.path.relpath(file_path, cwd)
            print(f'JSON file saved at: {rel_path}')

    # Save arrays separately as NPZ
    if save_arrays:
        file_path = os.path.join(save_path, f'{filename}_arrays.npz')
        np.savez_compressed(file_path, **arrays_dict)
        if print_log:
            cwd = os.getcwd()
            rel_path = os.path.relpath(file_path, cwd)
            print(f'Arrays saved at: {rel_path}')
            print('-' * 50)



def save_json_metadata(params, config, save_path, filename='metadata', save_arrays=True, print_log=True):
    # Split arrays and scalars
    params_dict = asdict(params)
    arrays_dict = {}
    for key, val in params_dict.items():
        if isinstance(val, np.ndarray):
            arrays_dict[key] = val
            params_dict[key] = f'np.ndarray with shape: {val.shape}'

    metadata = {
        'NetworkParams': params_dict,
        'config': config
    }
    # Save scalars as JSON
    file_path = os.path.join(save_path, f'{filename}.json')
    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=2)
    if print_log:
        title = " Saved Files "
        print(title.center(50, "-"))
        cwd = os.getcwd()
        rel_path = os.path.relpath(file_path, cwd)
        print(f'JSON file saved at: {rel_path}')

    # Save arrays separately as NPZ
    if save_arrays:
        file_path = os.path.join(save_path, f'{filename}_arrays.npz')
        np.savez_compressed(file_path, **arrays_dict)
        if print_log:
            cwd = os.getcwd()
            rel_path = os.path.relpath(file_path, cwd)
            print(f'Arrays saved at: {rel_path}')
            print('-' * 50)


def validate_save_path(save_path):
    """
    Validates and creates a save path if it doesn't exist.
    :param save_path: save path
    :return: No return value
    """
    try:
        os.makedirs(save_path, exist_ok=True)  # Creates if missing
        test_file = os.path.join(save_path, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise RuntimeError(f"Failed to validate save_path '{save_path}': {e}")





class ResultSaver:
    def __init__(self):
        self.results = []

    def add(self, result):
        """
        Adds result to the 'results' list
        :param result: Result to add
        :return: No return value
        """
        self.results.append(result)

    def save(self, path):
        """
        Saves results to pickle file
        :param path: Path inconfig filename
        :return: No return value
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)

    def reset(self):
        """
        Clears the 'results' list
        :return: No return value
        """
        self.results.clear()

    def summary(self):
        """
        Prints a summary of the available methods.
        """
        import inspect
        width = 50
        print("=" * width)
        print("ResultSaver Reference".center(width))
        print("=" * width)
        title = " Available Methods "
        print(title.center(width, "-"))
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue
            doc = method.__doc__
            if doc:
                # Split into lines and stop at :param, :return, etc.
                lines = doc.strip().splitlines()
                summary_lines = []
                for line in lines:
                    if line.strip().startswith(":"):  # stop at :param, :return, etc.
                        break
                    if line.strip():  # skip empty lines
                        summary_lines.append(line.strip())
                doc_summary = " ".join(summary_lines)
            else:
                doc_summary = "No description"
            print(f"{name:<10} : {doc_summary}")
        print("=" * width)






def load_results(path):
    """
    Loads results from pickle file
    :param path: Path inconfig filename
    :return: results list
    """
    import pickle
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results




# ------------------------------
# Functions for printing logs
# ------------------------------


def print_log(tag, message, width=15, pad_char='-'):
    """
    Print aligned log messages with a tag and padded dashes.

    Args:
        tag (str): The tag to appear in square brackets, e.g., 'CONFIG'
        message (str): The log message
        width (int): Total width reserved for the tag section (default: 15)
        pad_char (str): Character used to pad between tag and message (default: '-')
    """
    tag_str = f"[{tag}]"
    pad_len = max(0, width - len(tag_str))
    padding = f" {pad_char * pad_len} "
    print(f"{tag_str}{padding}{message}")





# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# Other helper functions not being used currently
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------



def generate_sensChannel_pairAPs(M_at, M_ar, tx_ap_position, rx_ap_position, target_positions, rcs_mean=10, uca_radius=0.05, wavelength=3e8/3e9,
                                 h_ap=10, h_t=1.5, rcs_random_state=None):
    rcs_rng = np.random.default_rng(rcs_random_state)
    N_t = target_positions.shape[0]
    rcs_mean = np.atleast_1d(rcs_mean)
    rcs_mean_vec = np.resize(rcs_mean, N_t)

    tx_ap_t_d2D = np.linalg.norm(tx_ap_position - target_positions, axis=1)
    tx_ap_t_d3D = np.sqrt(tx_ap_t_d2D ** 2 + (h_ap - h_t) ** 2)
    tx_ap_t_azimuth_deg = np.rad2deg(np.arctan2(target_positions[:, 1] - tx_ap_position[1], target_positions[:, 0] - tx_ap_position[0]))
    tx_ap_t_azimuth_deg = (tx_ap_t_azimuth_deg + 360) % 360
    tx_ap_t_elevation_deg = np.rad2deg(np.arctan2(tx_ap_t_d2D, h_ap - h_t))

    rx_ap_t_d2D = np.linalg.norm(rx_ap_position - target_positions, axis=1)
    rx_ap_t_d3D = np.sqrt(rx_ap_t_d2D ** 2 + (h_ap - h_t) ** 2)
    rx_ap_t_azimuth_deg = np.rad2deg(np.arctan2(target_positions[:, 1] - rx_ap_position[1], target_positions[:, 0] - rx_ap_position[0]))
    rx_ap_t_azimuth_deg = (rx_ap_t_azimuth_deg + 360) % 360
    rx_ap_t_elevation_deg = np.rad2deg(np.arctan2(rx_ap_t_d2D, h_ap - h_t))

    sensing_channel = np.zeros((N_t, M_ar, M_at), dtype=complex)
    for t in range(N_t):
        h_at = uca_array_response(M_a=M_at, uca_radius=uca_radius, wavelength=wavelength,
                                  azimuth_deg=tx_ap_t_azimuth_deg[t], elevation_deg=tx_ap_t_elevation_deg[t])
        h_tr = uca_array_response(M_a=M_ar, uca_radius=uca_radius, wavelength=wavelength,
                                  azimuth_deg=rx_ap_t_azimuth_deg[t], elevation_deg=rx_ap_t_elevation_deg[t])
        rcs_val = rcs_rng.exponential(scale=rcs_mean_vec[t])
        alpha = (np.sqrt((wavelength ** 2 * rcs_val) / ((4 * np.pi) ** 3 * tx_ap_t_d3D[t] ** 2 * rx_ap_t_d3D[t] ** 2)) *
                 np.exp(1j * rcs_rng.uniform(low=0, high=2 * np.pi)))
        sensing_channel[t, :, :] = alpha * (h_tr @ h_at.conj().T)

    return sensing_channel



# Old version
def get_target_rcs_v1(target_class='car', class_mean_rcs=None, sigma_aspect_db=None, target_velocity=None, vel_sigma_db=None,
                      vel_const=None, vel_vmax=None, vel_sigma_max_db=None, random_state=None):
    rng = np.random.default_rng(random_state)
    if class_mean_rcs is None:
        if target_class == 'car':
            class_mean_db = rng.uniform(low=10, high=20)
        elif target_class == 'human':
            class_mean_db = rng.uniform(low=-10, high=0)
        elif target_class == 'drone':
            class_mean_db = rng.uniform(low=-20, high=-5)
        else:
            raise ValueError("Invalid target class. Choose from 'car', 'human', or 'drone'.")
    else:
        class_mean_db = 10 * np.log10(class_mean_rcs)

    if sigma_aspect_db is None:
        if target_class == 'car':
            sigma_aspect_db = rng.uniform(low=1, high=3)
        elif target_class == 'human':
            sigma_aspect_db = rng.uniform(low=3, high=6)
        elif target_class == 'drone':
            sigma_aspect_db = rng.uniform(low=4, high=8)
    delta_aspect_db = rng.normal(loc=0, scale=sigma_aspect_db)

    if vel_sigma_db is None:
        if target_class == 'car':
            target_velocity = target_velocity if target_velocity is not None else rng.uniform(low=0, high=10)
            const = vel_const if vel_const is not None else 1.0
            vmax = vel_vmax if vel_vmax is not None else 20.0
            sigma_max_db = vel_sigma_max_db if vel_sigma_max_db is not None else 5.0
            vel_sigma_db = const * (target_velocity / vmax) * sigma_max_db
        elif target_class == 'human':
            target_velocity = target_velocity if target_velocity is not None else rng.uniform(low=0, high=1)
            const = vel_const if vel_const is not None else 1.0
            vmax = vel_vmax if vel_vmax is not None else 1.0
            sigma_max_db = vel_sigma_max_db if vel_sigma_max_db is not None else 1.0
            vel_sigma_db = const * (target_velocity / vmax) * sigma_max_db
        elif target_class == 'drone':
            target_velocity = target_velocity if target_velocity is not None else rng.uniform(low=0, high=10)
            const = vel_const if vel_const is not None else 1.0
            vmax = vel_vmax if vel_vmax is not None else 10.0
            sigma_max_db = vel_sigma_max_db if vel_sigma_max_db is not None else 5.0
            vel_sigma_db = const * (target_velocity / vmax) * sigma_max_db
    delta_vel_db = rng.normal(loc=0, scale=vel_sigma_db)

    rcs_db = class_mean_db + delta_aspect_db + delta_vel_db
    rcs = 10 ** (rcs_db / 10)

    return rcs



def get_sensLink_params_singleAP(ap_position, target_positions, target_velocities, h_ap=10, h_tg=1.5, target_classes=None,
                                 target_mean_rcs=None, prob_los_mode='3gpp', prev_measurements=None, return_df=False, random_state=None):
    # Distances
    ap_tg_d2D = np.linalg.norm(ap_position - target_positions, axis=1)
    ap_tg_d3D = np.sqrt(ap_tg_d2D ** 2 + (h_ap - h_tg) ** 2)
    N_tg = target_positions.shape[0]

    records = []
    for t in range(N_tg):
        prev_measurement = prev_measurements[t] if prev_measurements is not None else None
        los_prob = get_target_los_prob(ap_tg_d3D[t], mode=prob_los_mode, target_velocity=target_velocities[t], prev_measurement=prev_measurement)
        tg_class = target_classes[t] if target_classes is not None else 'car'
        tg_mean_rcs = target_mean_rcs[t] if target_mean_rcs is not None else None
        rcs = get_target_rcs(target_class=tg_class, class_mean_rcs=tg_mean_rcs, random_state=random_state)
        records.append({'target_id': t, 'target_distance': ap_tg_d3D[t], 'target_los_prob': los_prob, 'target_rcs_estimate': rcs})

    if return_df:
        sens_link_params_df = pd.DataFrame(records)
        return sens_link_params_df
    else:
        return records


# Old version
def extract_sensLink_features_v1(ap_positions, target_positions, relative_velocities, h_ap=10, h_tg=1.5, target_classes=None, target_mean_rcs=None,
                                 prob_los_mode='3gpp', prev_measurements=None, random_state=None):
    N_ap = ap_positions.shape[0]
    all_records = []
    for a in range(N_ap):
        sens_link_params = get_sensLink_params_singleAP(ap_position=ap_positions[a,:], target_positions=target_positions,
                                                        target_velocities=relative_velocities[a,:], h_ap=h_ap, h_tg=h_tg,
                                                        target_classes=target_classes,target_mean_rcs=target_mean_rcs,
                                                        prob_los_mode=prob_los_mode, prev_measurements=prev_measurements, random_state=random_state)
        for output in sens_link_params:
            all_records.append({'ap_id': a, **output})
    records_df = pd.DataFrame(all_records)
    return records_df



