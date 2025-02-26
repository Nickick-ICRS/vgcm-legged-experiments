import pandas as pd
import numpy as np
import optuna
import pinocchio as pin
import torch

import os

from legged_gym import LEGGED_GYM_ROOT_DIR

from vgcm.vgcm_ideal_model import VGCMIdealModel


optuna.logging.set_verbosity(optuna.logging.WARNING)

# TODO: Adjust these to something reasonable
MAX_XMAX = np.pi
MIN_XMIN = -np.pi

# Spring k_max <= K_MULT * k_min
K_MULT = 4


results_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'vgcm/experiment_results')
filepath = os.path.join(results_dir, 'basic_linear_experiment_results_0.csv')

df = pd.read_csv(filepath)

urdf_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/robots/pointfoot/WF_TRON1A/urdf/robot.urdf')
model = pin.buildModelFromUrdf(urdf_path)

frames = [
    'abad_L_Link', 'hip_L_Link', 'knee_L_Link', 'wheel_L_Link',
    'abad_R_Link', 'hip_R_Link', 'knee_R_Link', 'wheel_R_Link'
]

wheel_l_id = model.getJointId('wheel_L_Joint')
wheel_r_id = model.getJointId('wheel_R_Joint')


def compute_kinematics(q_pos, base_position, base_quaternion, tau):
    """
    Given joint positions and base pose, computes the end-effector position
    and joint base poses in the world frame.

    Args:
        q_pos: Array of joint values
        base_pos: [x, y, z] global position of the robot base
        base_quat: [qx, qy, qz, qw] quaternion orientation of the base
        tau: Array of joint torques

    Returns:
        ee_in_joint_frane: wheel pose in joint frames
        fs: equivalent force in gravity direction to create the torque tau
    """
    # Convert base quaternion to SE(3) transformation
    data = model.createData()
    xyz_quat = np.concatenate([base_position, base_quaternion])
    data.oMi[0] = pin.XYZQUATToSE3(xyz_quat)
    pin.forwardKinematics(model, data, q_pos)
    pin.updateFramePlacements(model, data)

    # Wheel global poses
    wheel_l_T = data.oMi[wheel_l_id]
    wheel_r_T = data.oMi[wheel_r_id]

    # Joint global poses
    fs = []
    g_dir_world = np.array([0, 0, -1])
    ee_in_joint_frame = []
    # Pinocchio idxs are our idx+1
    idxs = [1, 2, 3, 5, 6, 7]
    for idx in idxs:
        # Ignore the base joint
        joint_T = data.oMi[idx]
        if 'R_' in frames[idx-1]:
            ee_x = joint_T.inverse() * wheel_r_T
        else:
            ee_x = joint_T.inverse() * wheel_l_T
        ee_in_joint_frame.append(ee_x)
        # Tau = lambda (r x dir)
        gdir = joint_T.rotation.T @ g_dir_world
        rxd = np.cross(ee_x.translation, gdir)
        lam = tau[idx-1] * rxd / np.linalg.norm(rxd)
        f = lam * gdir
        fs.append(f)

    return ee_in_joint_frame, fs


def calculate_moment_arms():
    global df
    print("Calculating Moment Arms.")
    jp = df.filter(regex=r"^q\d+$").to_numpy()
    bp = df.filter(like="base_pos").to_numpy()
    bq = df.filter(like="base_quat").to_numpy()
    tau = df.filter(like="tau").to_numpy()

    cols_x = [f"{frame}_ee_pos" for frame in frames if "wheel" not in frame]
    cols_f = [f"{frame}_equiv_f" for frame in frames if "wheel" not in frame]
    _raw_data = np.zeros((len(df), len(cols_x)), dtype=object)
    _equiv_f = np.zeros((len(df), len(cols_f)), dtype=object)

    for i in range(len(df)):
        joint_poses, g_f = compute_kinematics(jp[i], bp[i], bq[i], tau[i])
        for j, jnt_SE3 in enumerate(joint_poses):
            _raw_data[i, j] = jnt_SE3.translation
            _equiv_f[i, j] = g_f[j]
    df_moment_arms = pd.DataFrame(_raw_data, columns=cols_x)
    df_forces = pd.DataFrame(_raw_data, columns=cols_f)
    df = pd.concat([df, df_moment_arms, df_forces], axis=1)
    print("Done")


def calculate_compensation_params(ee_xs, qs, axes, fs, k_min, k_max):
    """
    :param ee_xs Cartesian position of ee w.r.t joint parent (n, 3) m
    :param qs Joint current positions (n,) rads (for the relevant joint)
    :param axes Joint axes (n, 3)
    :param fs Force applied at ee in joint parent frame (n, 3) N
    :param k_min Minimum spring stiffness
    :param k_max Maximum spring stiffness
    """
    ee_xs = torch.tensor(ee_xs, dtype=torch.float32)
    qs = torch.tensor(qs, dtype=torch.float32)
    axes = torch.tensor(axes, dtype=torch.float32)
    fs = torch.tensor(fs, dtype=torch.float32)

    def get_tau(theta):
        c = torch.cos(theta).unsqueeze(-1)
        s = torch.sin(theta).unsqueeze(-1)
        r = c * ee_xs + s * torch.cross(axes, ee_xs, dim=1) + (1-c) * torch.sum(axes * ee_xs) * axes
        return torch.sum(torch.cross(r, fs, dim=1) * axes, dim=1)
    # Ideally we exactly compensate at theta = q
    preload = get_tau(torch.zeros_like(qs))

    dx = 0.01
    tau_0 = get_tau(torch.ones_like(qs) * -dx)
    tau_2 = get_tau(torch.ones_like(qs) * dx)
    # Gradient is spring constant
    sign = torch.ones_like(preload)
    sign[tau_2 < tau_0] = -1
    stiffness = torch.clip(np.abs((tau_2 - tau_0) / (2*dx)), k_min, k_max)
    zero_pos = sign * preload / stiffness

    return stiffness, zero_pos


calculate_moment_arms()
all_ee_xs = df.filter(like="_ee_pos")
all_ee_fs = df.filter(like="_equiv_f")
# Ignore joints 3 and 7 (wheels)
all_qs = df.filter(regex=r"^q(?!3$|7$)\d+$")


def get_axis(joint):
    name = joint.shortname()
    if 'RX' in name:
        return np.array([1, 0, 0])
    elif 'RY' in name:
        return np.array([0, 1, 0])
    elif 'RZ' in name:
        return np.array([0, 0, 1])
    else:
        return joint.extract().axis


idxs = [0, 1, 2, 4, 5, 6]
opt_n = 0


def objective(trial):
    k_min = trial.suggest_float("k_min", 0.01, 10_000)
    k_max = trial.suggest_float("k_max", k_min, K_MULT * k_min)
    x_min = trial.suggest_float("x_min", MIN_XMIN, MAX_XMAX)
    x_max = trial.suggest_float("x_max", x_min, MAX_XMAX)

    # Skip base joint -> +1
    axis = get_axis(model.joints[idxs[opt_n]+1])
    axes = np.stack([axis for _ in range(len(df))])

    base_name = frames[idxs[opt_n]]
    ee_xs = all_ee_xs[base_name+'_ee_pos']
    qs = all_qs[f'q{idxs[opt_n]}']
    fs = all_ee_fs[base_name+'_equiv_f']

    # Calculator forces k to be within k limits
    _k, x = calculate_compensation_params(
        ee_xs, qs, axes, fs, k_min, k_max)
    xma = torch.ones_like(x) * x_max
    xmi = torch.ones_like(x) * x_min
    reward_feasible = torch.sum(torch.ones_like(x)[(xmi < x) & (x < xma)])
    penalty_infeasible = -torch.sum(torch.ones_like(x)[(xmi > x) | (x > xma)] * 1e3)
    penalty_params = - (k_max - k_min) - (x_max - x_min)
    return reward_feasible + penalty_params + penalty_infeasible


def check_acc(result):
    k_min = result['k_min']
    k_max = result['k_max']
    x_min = result['x_min']
    x_max = result['x_max']

    # Skip base joint -> +1
    axis = get_axis(model.joints[idxs[opt_n]+1])
    axes = np.stack([axis for _ in range(len(df))])

    base_name = frames[idxs[opt_n]]
    ee_xs = all_ee_xs[base_name+'_ee_pos']
    qs = all_qs[f'q{idxs[opt_n]}']
    fs = all_ee_fs[base_name+'_equiv_f']

    # Calculator forces k to be within k limits
    _k, x = calculate_compensation_params(
        ee_xs, qs, axes, fs, k_min, k_max)
    xma = torch.ones_like(x) * x_max
    xmi = torch.ones_like(x) * x_min
    feasible = torch.sum(torch.ones_like(x)[(xmi < x) & (x < xma)])
    infeasible = torch.sum(torch.ones_like(x)[(xmi > x) | (x > xma)])
    print(f"Parameters {result} have {feasible} hits and {infeasible} misses")


results = []


def run_opt(i, idx):
    global opt_n
    print(f"Starting optimisation for joint {frames[idx]}.")
    opt_n = i
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    print("Done")
    results.append(study.best_params)


for i, idx in enumerate(idxs):
    run_opt(i, idx)

for i, idx in enumerate(idxs):
    print(f"{frames[idx]}:")
    check_acc(results[i])
    print(results[i])
