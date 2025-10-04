
import numpy as np
import pandas as pd
import src.utils.library as lib
import src.utils.visualization_utils as viz
from dataclasses import dataclass, field, fields
from typing import Union



@dataclass
class NetworkParams:
    # Structure parameters
    N_ap: int = field(default=10, metadata={"info": "Number of APs [default=10]"})
    M_a: int = field(default=16, metadata={"info": "Number of antennas per AP [default=16]"})
    N_RF: int = field(default=4, metadata={"info": "Number of RF chains per AP [default=4]"})
    N_cu: int = field(default=8, metadata={"info": "Number of comm users [default=8]"})
    N_tg: int = field(default=6, metadata={"info": "Number of sensing targets [default=6]"})
    h_ap: float = field(default=10.0, metadata={"info": "Height of APs in meters [default=10.0]"})
    h_cu: float = field(default=1.5, metadata={"info": "Height of comm users in meters [default=1.5]"})
    h_tg: float = field(default=1.5, metadata={"info": "Height of sensing targets in meters [default=1.5]"})

    # Environment dimensions
    env_x: int = field(default=1000, metadata={"info": "Environment size in x-dimension in meters [default=1000]"})
    env_y: int = field(default=1000, metadata={"info": "Environment size in y-dimension in meters [default=1000]"})

    # Minimum distances (in meters)
    min_ap_dist: int = field(default=200, metadata={"info": "Minimum distance b/w APs in meters [default=200]"})
    min_user_dist: int = field(default=5, metadata={"info": "Minimum distance b/w comm users in meters [default=5]"})
    min_user2ap_dist: int = field(default=10, metadata={"info": "Minimum distance b/w comm users and APs in meters [default=10]"})
    min_target_dist: int = field(default=10, metadata={"info": "Minimum distance b/w sensing targets in meters [default=10]"})
    min_target2user_ap_dist: int = field(default=5, metadata={"info": "Minimum distance b/w sensing targets and comm users/APs in meters [default=5]"})

    # Position modes
    ap_position_mode: str = field(default='uniform', metadata={"info": "AP position mode from 'uniform', 'circle', 'line' [default='uniform']"})
    user_position_mode: str = field(default='uniform', metadata={"info": "Comm user position mode from 'uniform', 'circle', 'line' [default='uniform']"})
    target_position_mode: str = field(default='uniform', metadata={"info": "Sensing target position mode from 'uniform', 'circle', 'line' [default='uniform']"})
    ap_circle_radius: int = field(default=None, metadata={"info": "If ap_position_mode is 'circle', the circle radius in meters [default=None]"})
    ap_line_point1: tuple = field(default=None, metadata={"info": "If ap_position_mode is 'line', the line start point (x1, y1) in meters [default=None]"})
    ap_line_point2: tuple = field(default=None, metadata={"info": "If ap_position_mode is 'line', the line end point (x2, y2) in meters [default=None]"})
    user_circle_radius: int = field(default=None, metadata={"info": "If user_position_mode is 'circle', the circle radius in meters [default=None]"})
    user_line_point1: tuple = field(default=None, metadata={"info": "If user_position_mode is 'line', the line start point (x1, y1) in meters [default=None]"})
    user_line_point2: tuple = field(default=None, metadata={"info": "If user_position_mode is 'line', the line end point (x2, y2) in meters [default=None]"})
    target_circle_radius: int = field(default=None, metadata={"info": "If target_position_mode is 'circle', the circle radius in meters [default=None]"})
    target_line_point1: tuple = field(default=None, metadata={"info": "If target_position_mode is 'line', the line start point (x1, y1) in meters [default=None]"})
    target_line_point2: tuple = field(default=None, metadata={"info": "If target_position_mode is 'line', the line end point (x2, y2) in meters [default=None]"})

    # Carrier parameters
    fc: float = field(default=3.0e9, metadata={"info": "Carrier frequency in Hz [default=3.0e9]"})
    c: float = field(default=3e8, metadata={"info": "Speed of light in m/s [default=3e8]"})

    @property
    def wavelength(self):
        """Wavelength in meters [c/fc]"""
        return self.c / self.fc

    @property
    def uca_radius(self):
        """Radius of UCA in meters [wavelength/2]"""
        return self.wavelength / 2

    # Target parameters
    target_velocities: Union[list, np.ndarray] = field(default=None, metadata={"info": "(N_ap, N_tg) target velocities in m/s [init=0 if None]"})
    target_mean_rcs: Union[list, np.ndarray] = field(default=None, metadata={"info": "(N_tg,) target mean RCS in linear scale (m2) [init=10 if None]"})
    target_classes: Union[list, np.ndarray] = field(default=None, metadata={"info": "(N_tg,) list of target classes, e.g., 'car', 'human', 'drone' [init='car' if None]"})
    target_prob_los_mode: str = field(default='3gpp', metadata={"info": "Target probability of LOS mode from '3gpp', 'logistic' [default=3gpp]"})
    target_prev_measurements: Union[list, np.ndarray] = field(default=None, metadata={"info": "(N_ap, N_tg) previous target SNR for logistic LOS probability [default=None]"})

    # Random states
    ap_position_random_state: int = field(default=None, metadata={"info": "Random state for AP positions [default=None]"})
    user_position_random_state: int = field(default=None, metadata={"info": "Random state for comm user positions [default=None]"})
    target_position_random_state: int = field(default=None, metadata={"info": "Random state for sensing target positions [default=None]"})
    user_channel_random_state: int = field(default=None, metadata={"info": "Random state for comm user channels [default=None]"})
    target_channel_random_state: int = field(default=None, metadata={"info": "Random state for sensing target channels [default=None]"})

    def __post_init__(self):
        if self.target_velocities is None:
            self.target_velocities = np.zeros((self.N_ap, self.N_tg))
        else:
            self.target_velocities = np.array(self.target_velocities)
            if self.target_velocities.shape != (self.N_ap, self.N_tg):
                raise ValueError("Shape of target_velocities must be (N_ap, N_tg).")

        if self.target_mean_rcs is None:
            self.target_mean_rcs = np.ones(self.N_tg) * 10
        else:
            self.target_mean_rcs = np.array(self.target_mean_rcs)
            if len(self.target_mean_rcs) != self.N_tg:
                raise ValueError("Length of target_mean_rcs must be equal to N_tg.")

        if self.target_classes is None:
            self.target_classes = ['car' for _ in range(self.N_tg)]

    def summary(self, show_functions: bool = True):
        """
        Print a summary of the parameters and available methods.
        """
        import inspect
        width = 50
        print("=" * width)
        print("NetworkParams Reference".center(width))
        print("=" * width)
        for f in fields(self):
            info = f.metadata.get("info", "")
            print(f"{f.name:<25} : {info}")
        title = " Properties "
        print(title.center(width, "="))
        for name, attr in inspect.getmembers(self.__class__):
            if isinstance(attr, property):
                doc = attr.__doc__.strip() if attr.__doc__ else "No description"
                print(f"{name:<25} : {doc}")
        if show_functions:
            title = " Available Methods "
            print(title.center(width, "="))
            for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
                if name.startswith("_"):  # skip internals
                    continue
                doc = method.__doc__.strip() if method.__doc__ else "No description"
                print(f"{name:<25} : {doc}")
        print("=" * width)












class NetworkEnvironment:
    _attr_metadata = {
        "params": "NetworkParams instance containing network parameters",
        "ap_positions": "AP positions (N_ap x 2)",
        "user_positions": "Comm user positions (N_cu x 2)",
        "target_positions": "Sensing target positions (N_tg x 2)",
        "commLinks_df": "DataFrame containing communication link parameters",
        "sensLinks_df": "DataFrame containing sensing link parameters",
        "G_sens": "Sensing channel gains matrix (N_ap x N_ap x N_tg)",
    }

    def __init__(self, params: NetworkParams):
        self.params = params
        self.ap_positions = None
        self.user_positions = None
        self.target_positions = None
        self.commLinks_df = None
        self.sensLinks_df = None
        self.G_sens = None


    def generate_topology(self):
        """Generate AP, user, and target positions."""
        p = self.params
        if self.ap_positions is None:
            self.ap_positions = lib.generate_AP_positions(N_ap=p.N_ap, env_x=p.env_x, env_y=p.env_y, mode=p.ap_position_mode,min_dist=p.min_ap_dist,
                                                          circle_radius=p.ap_circle_radius, line_point1=p.ap_line_point1, line_point2=p.ap_line_point2,
                                                          random_state=p.ap_position_random_state)
        if self.user_positions is None:
            self.user_positions = lib.generate_entity_positions(N_ent=p.N_cu, env_x=p.env_x, env_y=p.env_y, mode=p.user_position_mode, min_entity_dist=p.min_user_dist,
                                                                min_entity2current_dist=p.min_user2ap_dist, current_positions=self.ap_positions, circle_radius=p.user_circle_radius,
                                                                line_point1=p.user_line_point1, line_point2=p.user_line_point2, random_state=p.user_position_random_state)
        if self.target_positions is None:
            current_positions = np.vstack([self.ap_positions, self.user_positions])
            self.target_positions = lib.generate_entity_positions(N_ent=p.N_tg, env_x=p.env_x, env_y=p.env_y, mode=p.target_position_mode, min_entity_dist=p.min_target_dist,
                                                                  min_entity2current_dist=p.min_target2user_ap_dist, current_positions=current_positions, circle_radius=p.target_circle_radius,
                                                                  line_point1=p.target_line_point1, line_point2=p.target_line_point2, random_state=p.target_position_random_state)


    def set_position(self, obj_type=None, obj_idx=None, coords=None):
        """
        Set the position of an object.
        :param obj_type: Type of object. Must be 'AP', 'user', or 'target'.
        :param obj_idx: Index of object.
        :param coords: Tuple of coordinates (x, y).
        :return: No return value.
        """
        if self.ap_positions is None or self.user_positions is None:
            raise ValueError("Topology not generated. Call generate_topology() first.")
        if coords is None:
            raise ValueError("Coordinates (x, y) must be specified.")
        if obj_type == 'AP':
            if obj_idx >= self.params.N_ap:
                raise ValueError("AP index out of range.")
            self.ap_positions[obj_idx, :] = coords
        elif obj_type == 'user':
            if obj_idx >= self.params.N_cu:
                raise ValueError("User index out of range.")
            self.user_positions[obj_idx, :] = coords
        elif obj_type == 'target':
            if obj_idx >= self.params.N_tg:
                raise ValueError("Target index out of range.")
            self.target_positions[obj_idx, :] = coords
        else:
            raise ValueError("Invalid object type. Must be 'AP', 'user', or 'target'.")


    def update_positions(self, obj_type=None):
        """
        Update the positions of APs, users, or targets.
        :param obj_type: Type of object. Must be 'AP', 'user', or 'target'.
        :return: No return value.
        """
        obj_types = ['all', 'AP', 'user', 'target', 'AP-user', 'AP-target', 'user-target']
        if obj_type not in obj_types:
            raise ValueError(f"Invalid object type. Must be one of {obj_types}.")
        p = self.params
        if obj_type == 'all' or obj_type == 'AP' or obj_type == 'AP-user' or obj_type == 'AP-target':
            self.ap_positions = lib.generate_AP_positions(N_ap=p.N_ap, env_x=p.env_x, env_y=p.env_y, mode=p.ap_position_mode,min_dist=p.min_ap_dist,
                                                          circle_radius=p.ap_circle_radius, line_point1=p.ap_line_point1, line_point2=p.ap_line_point2,
                                                          random_state=p.ap_position_random_state)
        if obj_type == 'all' or obj_type == 'user' or obj_type == 'user-target' or obj_type == 'AP-user':
            if self.ap_positions is None:
                raise ValueError("AP positions not set. Call generate_topology() first.")
            self.user_positions = lib.generate_entity_positions(N_ent=p.N_cu, env_x=p.env_x, env_y=p.env_y, mode=p.user_position_mode, min_entity_dist=p.min_user_dist,
                                                                min_entity2current_dist=p.min_user2ap_dist, current_positions=self.ap_positions, circle_radius=p.user_circle_radius,
                                                                line_point1=p.user_line_point1, line_point2=p.user_line_point2, random_state=p.user_position_random_state)
        if obj_type == 'all' or obj_type == 'target' or obj_type == 'AP-target' or obj_type == 'user-target':
            if self.ap_positions is None or self.user_positions is None:
                raise ValueError("AP or user positions not set. Call generate_topology() first.")
            current_positions = np.vstack([self.ap_positions, self.user_positions])
            self.target_positions = lib.generate_entity_positions(N_ent=p.N_tg, env_x=p.env_x, env_y=p.env_y, mode=p.target_position_mode, min_entity_dist=p.min_target_dist,
                                                                  min_entity2current_dist=p.min_target2user_ap_dist, current_positions=current_positions, circle_radius=p.target_circle_radius,
                                                                  line_point1=p.target_line_point1, line_point2=p.target_line_point2, random_state=p.target_position_random_state)

    def plot_topology(self, figsize=(8, 7), show='all', margin=50, equal_aspect=False):
        """
        Plots the network topology.
        :param figsize: Figure size (default=(8, 7))
        :param show: What to show: 'all', 'AP', 'CU', 'T', etc (default='all')
        :param margin: Margin around the environment box (default=50)
        :param equal_aspect: Whether to set equal aspect ratio (default=False)
        :return: No return value
        """
        if self.ap_positions is None or self.user_positions is None or self.target_positions is None:
            raise ValueError("Topology not generated. Call generate_topology() first.")
        viz.plot_topology(self.ap_positions, self.user_positions, self.target_positions, self.params.env_x, self.params.env_y, show=show, figsize=figsize, margin=margin, equal_aspect=equal_aspect)


    def generate_commChannels(self):
        """
        Generates communication link parameters in commLinks_df and channel matrix H_comm.
        :return: Channel matrix H_comm of shape (N_ap, N_cu, M_a)
        """
        p = self.params
        if self.ap_positions is None or self.user_positions is None:
            raise ValueError("Topology not generated. Call generate_topology() first.")
        df_list = []
        H_comm = np.zeros((p.N_ap, p.N_cu, p.M_a), dtype=complex)
        for ap_idx in range(p.N_ap):
            H_a, df = lib.generate_commChannel_singleAP(M_a=p.M_a, ap_position=self.ap_positions[ap_idx, :], user_positions=self.user_positions, uca_radius=p.uca_radius, h_ap=p.h_ap,
                                                        h_cu=p.h_cu, fc=p.fc, c=p.c, mode='3gpp', return_df=True, ch_random_state=p.user_channel_random_state)
            df.insert(0, "ap_id", ap_idx)
            df_list.append(df)
            H_comm[ap_idx, :, :] = H_a
        self.commLinks_df = pd.concat(df_list, ignore_index=True)
        return H_comm


    def generate_commLink_features(self, normalize=False):
        """
        Generates communication link parameters and channels and then extract link features G_comm and S_comm from channel matrix H_comm.
        :param normalize: Whether to normalize the user channel vectors (default=False)
        :return: G_mat: Channel gains (N_ap x N_cu) and S_mat: Spatial correlations (N_ap x N_cu x N_cu)
        """
        H_comm = self.generate_commChannels()
        G_comm, S_comm = lib.extract_commLink_features(H_comm, normalize=normalize)
        return G_comm, S_comm


    def generate_sensLink_params(self):
        """
        Generates sensing link parameters in sensLinks_df.
        :return: Returns sensLinks_df DataFrame with columns:
        'ap_id', 'target_id', 'target_distance', 'target_azimuth_deg', 'target_elevation_deg', 'target_los_prob', 'target_rcs_estimate'
        """
        sensLinks_list = []
        p = self.params
        if self.ap_positions is None or self.target_positions is None:
            raise ValueError("Topology not generated. Call generate_topology() first.")
        for ap_idx in range(p.N_ap):
            ap_tg_d2D = np.linalg.norm(self.ap_positions[ap_idx, :] - self.target_positions, axis=1)
            ap_tg_d3D = np.sqrt(ap_tg_d2D**2 + (p.h_ap - p.h_tg)**2)
            ap_tg_azimuth_deg = np.rad2deg(
                np.arctan2(self.target_positions[:, 1] - self.ap_positions[ap_idx, 1], self.target_positions[:, 0] - self.ap_positions[ap_idx, 0]))
            ap_tg_azimuth_deg = (ap_tg_azimuth_deg + 360) % 360
            ap_tg_elevation_deg = np.rad2deg(np.arctan2(ap_tg_d2D, p.h_ap - p.h_tg))
            for tg_idx in range(p.N_tg):
                prev_measurement = p.target_prev_measurements[ap_idx, tg_idx] if p.target_prev_measurements is not None else None
                los_prob = lib.get_target_los_prob(ap_tg_d3D[tg_idx], mode=p.target_prob_los_mode, target_velocity=p.target_velocities[ap_idx,tg_idx], prev_measurement=prev_measurement)
                rcs = lib.get_target_rcs(target_class=p.target_classes[tg_idx], class_mean_rcs=p.target_mean_rcs[tg_idx], target_velocity=p.target_velocities[ap_idx,tg_idx],
                                         v_ref=1.0, random_state=p.target_channel_random_state)
                sensLinks_list.append({'ap_id': ap_idx, 'target_id': tg_idx, 'target_distance': ap_tg_d3D[tg_idx], 'target_azimuth_deg': ap_tg_azimuth_deg[tg_idx],
                                       'target_elevation_deg': ap_tg_elevation_deg[tg_idx], 'target_los_prob': los_prob, 'target_rcs_estimate': rcs})
        self.sensLinks_df = pd.DataFrame(sensLinks_list)
        return self.sensLinks_df



    def generate_sensLink_features(self, update_sensLinks_df=True, dB_scale=False, normalize_losWeight=True, w_los=1.0, w_nlos=0.2):
        """
        Generates sensing link parameters and then extract link features G_sens from sensLinks_df.
        :param update_sensLinks_df: Whether to update sensLinks_df by calling generate_sensLink_params() again (default=True)
        :param dB_scale: Whether to return G_sens in dB scale (default=False)
        :param normalize_losWeight: Whether to normalize the LOS weight (default=True)
        :param w_los: Weight for LOS link (default=1.0)
        :param w_nlos: Weight for non-LOS link (default=0.2)
        :return: Matrix G_sens with sensing channel gains (N_ap x N_ap x N_tg)
        """
        p = self.params
        if update_sensLinks_df:
            self.sensLinks_df = self.generate_sensLink_params()
        sdf = self.sensLinks_df
        w_max = w_nlos + (w_los - w_nlos) * (np.max(sdf['target_los_prob']))

        G_sens = np.zeros((p.N_ap, p.N_ap, p.N_tg))
        for atx in range(p.N_ap):
            for arx in range(p.N_ap):
                for tg in range(p.N_tg):
                    # atx to tg
                    d_atxtg = sdf.loc[(sdf['ap_id'] == atx) & (sdf['target_id'] == tg), 'target_distance'].values[0]
                    rcs_atxtg = sdf.loc[(sdf['ap_id'] == atx) & (sdf['target_id'] == tg), 'target_rcs_estimate'].values[0]
                    plos_atxtg = sdf.loc[(sdf['ap_id'] == atx) & (sdf['target_id'] == tg), 'target_los_prob'].values[0]
                    w_atxtg = w_nlos + (w_los - w_nlos) * plos_atxtg
                    w_atxtg = w_atxtg / w_max if normalize_losWeight else w_atxtg

                    # tg to arx
                    d_tgarx = sdf.loc[(sdf['ap_id'] == arx) & (sdf['target_id'] == tg), 'target_distance'].values[0]
                    rcs_tgarx = sdf.loc[(sdf['ap_id'] == arx) & (sdf['target_id'] == tg), 'target_rcs_estimate'].values[0]
                    plos_tgarx = sdf.loc[(sdf['ap_id'] == arx) & (sdf['target_id'] == tg), 'target_los_prob'].values[0]
                    w_tgarx = w_nlos + (w_los - w_nlos) * plos_tgarx
                    w_tgarx = w_tgarx / w_max if normalize_losWeight else w_tgarx

                    g_txtgrx = (1 / (d_atxtg ** 2)) * (1 / (d_tgarx ** 2)) * (
                                (rcs_atxtg + rcs_tgarx) / 2) * w_atxtg * w_tgarx
                    G_sens[atx, arx, tg] = g_txtgrx
        # Free-space pathloss factor
        self.G_sens = G_sens * ((p.wavelength ** 2) / ((4 * np.pi) ** 3))
        if dB_scale:
            G_sens_db = 10 * np.log10(self.G_sens)
            return G_sens_db
        else:
            return self.G_sens


    def generate_sensChannel_pairAPs(self, tx_ap_idx, rx_ap_idx, update_G_sens=True, update_sensLinks_df=True, normalize_losWeight=True, w_los=1.0, w_nlos=0.2):
        """
        Generate sensing channel matrix for a pair of APs to all targets.
        :param tx_ap_idx: Index of the transmitting AP.
        :param rx_ap_idx: Index of the receiving AP.
        :param update_G_sens: Whether to update G_sens by calling generate_sensLink_features() again (default=True)
        :param update_sensLinks_df: Input to generate_sensLink_features() for when update_G_sens is True (default=True)
        :param normalize_losWeight: Input to generate_sensLink_features() for when update_G_sens is True (default=True)
        :param w_los: Input to generate_sensLink_features() for when update_G_sens is True (default=1.0)
        :param w_nlos: Input to generate_sensLink_features() for when update_G_sens is True (default=0.2)
        :return: Sensing channel matrix H_sens_pair of shape (N_tg, M_a, M_a)
        """
        p = self.params
        sdf = self.sensLinks_df
        if self.ap_positions is None or self.target_positions is None:
            raise ValueError("Topology not generated. Call generate_topology() first.")
        if tx_ap_idx >= p.N_ap or rx_ap_idx >= p.N_ap:
            raise ValueError("AP index out of range.")
        if update_G_sens:
            self.generate_sensLink_features(update_sensLinks_df=update_sensLinks_df, normalize_losWeight=normalize_losWeight, w_los=w_los, w_nlos=w_nlos)

        H_sens_pair = np.zeros((p.N_tg, p.M_a, p.M_a), dtype=complex)
        for tg_idx in range(p.N_tg):
            tx_az_deg = sdf.loc[(sdf['ap_id'] == tx_ap_idx) & (sdf['target_id'] == tg_idx), 'target_azimuth_deg'].values[0]
            tx_el_deg = sdf.loc[(sdf['ap_id'] == tx_ap_idx) & (sdf['target_id'] == tg_idx), 'target_elevation_deg'].values[0]
            h_at = lib.uca_array_response(M_a=p.M_a, uca_radius=p.uca_radius, wavelength=p.wavelength, azimuth_deg=tx_az_deg, elevation_deg=tx_el_deg)

            rx_az_deg = sdf.loc[(sdf['ap_id'] == rx_ap_idx) & (sdf['target_id'] == tg_idx), 'target_azimuth_deg'].values[0]
            rx_el_deg = sdf.loc[(sdf['ap_id'] == rx_ap_idx) & (sdf['target_id'] == tg_idx), 'target_elevation_deg'].values[0]
            h_tr = lib.uca_array_response(M_a=p.M_a, uca_radius=p.uca_radius, wavelength=p.wavelength, azimuth_deg=rx_az_deg, elevation_deg=rx_el_deg)

            ch_gain = self.G_sens[tx_ap_idx, rx_ap_idx, tg_idx]
            H_sens_pair[tg_idx, :, :] = np.sqrt(ch_gain) * (h_tr @ h_at.conj().T)

        return H_sens_pair


    def summary(self):
        """
        Print a summary of the attributes and available methods.
        """
        import inspect
        width = 50
        print("=" * width)
        print("NetworkEnvironment Reference".center(width))
        print("=" * width)
        title = " Available Attributes "
        print(title.center(width, "-"))
        for attr_name, val in self.__dict__.items():
            if not attr_name.startswith("_"):
                info = self._attr_metadata.get(attr_name, "No description")
                print(f"{attr_name:<20} : {info}")
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

            print(f"{name:<30} : {doc_summary}")
        print("=" * width)

































