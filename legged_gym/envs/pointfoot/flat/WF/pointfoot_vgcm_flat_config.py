from legged_gym.envs import PointFootFlatCfg, PointFootFlatCfgPPO
from vgcm.vgcm_ideal_model import K_MAX, K_MIN, X_MAX, X_MIN


class PointFootVGCMFlatCfg(PointFootFlatCfg):
    class env(PointFootFlatCfg.env):
        # Plus 2 alphas + 6 ks + 6 xs + 12 more actions
        num_privileged_obs = 31 + 12 + 2 + 6 + 6
        num_propriceptive_obs = 31 + 12 + 2 + 6 + 6
        num_actions = 20

    class init_state(PointFootFlatCfg.init_state):
        k_min = K_MIN
        k_max = K_MAX
        x_min = X_MIN
        x_max = X_MAX
        k = (K_MIN + K_MAX) / 2.
        default_vgcm_stiffnesses = {  # target stiffness when action = 0.0
            "abad_L_Joint": k[0],
            "hip_L_Joint": k[1],
            "knee_L_Joint": k[2],
            "abad_R_Joint": k[3],
            "hip_R_Joint": k[4],
            "knee_R_Joint": k[5],
        }
        alpha_range = (1., 60.)

    class control(PointFootFlatCfg.control):
        control_type = "P_AND_V" # P: position, V: velocity, T: torques. 
                                 # P_AND_V: some joints use position control 
                                 # and others use vecocity control.
        # PD Drive parameters:
        stiffness = {
            "abad_L_Joint": 40,
            "hip_L_Joint": 40,
            "knee_L_Joint": 40,
            "abad_R_Joint": 40,
            "hip_R_Joint": 40,
            "knee_R_Joint": 40,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 1.8,
            "hip_L_Joint": 1.8,
            "knee_L_Joint": 1.8,
            "abad_R_Joint": 1.8,
            "hip_R_Joint": 1.8,
            "knee_R_Joint": 1.8,
            "wheel_L_Joint": 0.5,
            "wheel_R_Joint": 0.5,
        }  # [N*m*s/rad]
        # action scale: target angle = actionscale * action + defaultangle
        # action_scale_pos is the action scale of joints that use position control
        # action_scale_vel is the action scale of joints that use velocity control
        action_scale_pos = 0.25
        action_scale_vel = 8
        action_scale_stiffness = (K_MAX - K_MIN) / 2.
        action_scale_equilibrium = (X_MAX - X_MIN) / 2.
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class noise(PointFootFlatCfg.noise):
        class noise_scales(PointFootFlatCfg.noise.noise_scales):
            vgcm_k = 0.05
            vgcm_x = 0.001

    class normalization(PointFootFlatCfg.normalization):
        class obs_scales(PointFootFlatCfg.normalization.obs_scales):
            vgcm_k = 2 / (K_MAX - K_MIN)
            vgcm_x = 2 / (X_MAX - X_MIN)

    class rewards(PointFootFlatCfg.rewards):
        class scales(PointFootFlatCfg.rewards.scales):
            # base class
            action_rate = -0.01
            ang_vel_xy = 0.0 # off
            base_height = -20.0
            collision = -10.0
            dof_acc = -2.5e-07
            dof_pos_limits = -2.0
            dof_vel = 0.0 # off
            feet_air_time = 0.0 # off
            feet_contact_forces = 0.0 # off
            feet_stumble = 0.0 # off
            lin_vel_z = 0.0 # off
            no_fly = 1.0
            orientation = -5.0 # 很重要，不加的话会导致存活时间下降
            stand_still = -1.0
            termination = 0.0 # off
            torque_limits = 0.0 # off
            torques = -2.5e-05
            tracking_ang_vel = 5.0 # off
            tracking_lin_vel = 10.0
            unbalance_feet_air_time = 0.0 # off
            unbalance_feet_height = 0.0 # off
            feet_distance = 5 # -100
            survival = 0.1
            wheel_adjustment = 1.0 # 1.0 off
            inclination = 0.0 # off
            # new added
            # VGCM_stiffness = -1.0
            # VGCM_equilibrium = -1.0

        base_height_target = 0.65
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        min_feet_distance = 0.29
        max_feet_distance = 0.32
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        base_height_target = 0.65 + 0.1664
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001

class PointFootVGCMFlatCfgPPO(PointFootFlatCfgPPO):
    class policy(PointFootFlatCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]

    class runner(PointFootFlatCfgPPO.runner):
        experiment_name = 'pointfoot_flat_vgcm'
        max_iterations = 10000
