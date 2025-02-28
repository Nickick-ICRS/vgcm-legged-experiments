import pandas as pd
import numpy as np
import optuna
import pinocchio as pin

import os

from legged_gym import LEGGED_GYM_ROOT_DIR


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


def get_axis(joint):
    """
    Extract joint axis from a pinocchio joint
    """
    name = joint.shortname()
    if 'RX' in name:
        return np.array([1, 0, 0])
    elif 'RY' in name:
        return np.array([0, 1, 0])
    elif 'RZ' in name:
        return np.array([0, 0, 1])
    else:
        return joint.extract().axis


def compute_ideal_stiffnesses(q_pos, q_vel, tau, base_position, base_quaternion):
    """
    Given joint positions and torques, computes the ideal spring stiffness
    to linearly compensate for system forces

    Args:
        q_pos: Array of joint positions
        q_vel: Array of joint velocities
        tau: Array of joint torques
        base_position: Position of the base link in the global frame
        base_quaternion: Orientation of the base link in the global frame

    Returns:
        k: Array of stiffnesses for non-wheel joints
    """
    # Convert base quaternion to SE(3) transformation
    data = model.createData()
    xyz_quat = np.concatenate([base_position, base_quaternion])
    data.oMi[0] = pin.XYZQUATToSE3(xyz_quat)
    pin.forwardKinematics(model, data, q_pos)
    pin.updateFramePlacements(model, data)

    # Joint global poses
    pin.computeJointJacobians(model, data, q_pos)
    J_l = pin.getFrameJacobian(model, data, wheel_l_id, pin.LOCAL)
    J_r = pin.getFrameJacobian(model, data, wheel_r_id, pin.LOCAL)
    # TODO: Consider velocity?
    pin.computeJointJacobiansTimeVariation(model, data, q_pos, q_vel)
    dJ_l = pin.getFrameJacobianTimeVariation(model, data, wheel_l_id, pin.LOCAL)
    dJ_r = pin.getFrameJacobianTimeVariation(model, data, wheel_r_id, pin.LOCAL)

    k = dJ_l.T @ J_l @ tau + dJ_r.T @ J_r @ tau
    # Pinocchio idxs are our idx+1
    idxs = [1, 2, 3, 5, 6, 7]

    return k[idxs]


def build_stiffness_df():
    global df
    print("Building Ideal Stiffness Database.")
    jp = df.filter(regex=r"^q\d+$").to_numpy()
    jv = df.filter(like="dq").to_numpy()
    tau = df.filter(like="tau").to_numpy()
    base_pos = df.filter(like="base_pos").to_numpy()
    base_quat = df.filter(like="base_quat").to_numpy()

    cols_k = [f"{frame}_ideal_k" for frame in frames if "wheel" not in frame]
    _raw_data = np.zeros((len(df), len(cols_k)), dtype=object)

    for i in range(len(df)):
        stiffnesses = compute_ideal_stiffnesses(jp[i], jv[i], tau[i], base_pos[i], base_quat[i])
        for j, k in enumerate(stiffnesses):
            _raw_data[i, j] = k
    df_k = pd.DataFrame(_raw_data, columns=cols_k)
    df = pd.concat([df, df_k], axis=1)
    print("Done")


def calculate_compensation_params(ideal_ks, q, tau, k_min, k_max):
    """
    :param ideal_ks Ideal spring stiffnesses (i.e. actual gradient of torque function at q)
    :param q Joint current positions (for the relevant joint)
    :param tau Joint torques (for the relevant joint)
    :param k_min Minimum spring stiffness
    :param k_max Maximum spring stiffness
    """
    stiffness = np.clip(np.abs(ideal_ks), k_min, k_max)
    zero_pos = q - tau / stiffness

    return stiffness, zero_pos


build_stiffness_df()
ideal_ks = df.filter(like="_ideal_k")
all_q = df.filter(regex=r"^q\d+$")
all_tau = df.filter(like="tau")

idxs = [0, 1, 2, 4, 5, 6]
opt_n = 0


def objective(trial):
    k_min = trial.suggest_float("k_min", 0.01, 10_000)
    k_max = k_min * 4.  # trial.suggest_float("k_max", k_min, K_MULT * k_min)
    x_min = trial.suggest_float("x_min", MIN_XMIN, 0)
    x_max = trial.suggest_float("x_max", 0, MAX_XMAX)

    base_name = frames[idxs[opt_n]]
    ideal_k = ideal_ks[base_name+'_ideal_k']
    q = all_q[f'q{idxs[opt_n]}']
    tau = all_tau[f'tau{idxs[opt_n]}']

    # Calculator forces k to be within k limits
    k, x = calculate_compensation_params(
        ideal_k, q, tau, k_min, k_max)
    xma = np.ones_like(x) * x_max
    xmi = np.ones_like(x) * x_min
    # Assume xmin/max scale with stiffness
    xma = xma * k_min / k
    xmi = xmi * k_min / k
    reward_feasible = np.sum(np.ones_like(x)[(xmi < x) & (x < xma)])
    penalty_infeasible = - np.sum(np.ones_like(x)[(xmi > x) | (x > xma)] * 1e2)
    penalty_params = - (k_max - k_min) - (x_max - x_min)
    return reward_feasible + penalty_params + penalty_infeasible


def check_acc(result, idx):
    k_min = result['k_min']
    k_max = 4. * k_min  # result['k_max']
    x_min = result['x_min']
    x_max = result['x_max']

    base_name = frames[idx]
    ideal_k = ideal_ks[base_name+'_ideal_k']
    q = all_q[f'q{idx}']
    tau = all_tau[f'tau{idx}']

    # Calculator forces k to be within k limits
    k, x = calculate_compensation_params(
        ideal_k, q, tau, k_min, k_max)
    xma = np.ones_like(x) * x_max
    xmi = np.ones_like(x) * x_min
    # Assume xmin/max scale with stiffness
    xma = xma * k_min / k
    xmi = xmi * k_min / k
    feasible = np.sum(np.ones_like(x)[(xmi < x) & (x < xma)])
    infeasible = np.sum(np.ones_like(x)[(xmi > x) | (x > xma)])
    reward = feasible - 1e2 * infeasible - (k_max - k_min) - (x_max - x_min)
    print(f"Parameters {result} have {feasible} hits and {infeasible} misses (reward: {reward})")


results_raw = []
results = {}


def run_opt(i, idx):
    global opt_n
    print(f"Starting optimisation for joint {frames[idx]}.")
    opt_n = i
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200, show_progress_bar=True)
    print("Done")
    results_raw.append(study.best_params)
    for param in study.best_params.keys():
        results[f"{param}{i}"] = [study.best_params[param]]
    results[f"k_max{i}"] = (study.best_params['k_min'] * 4.)


for i, idx in enumerate(idxs):
    run_opt(i, idx)

df_frames_params = pd.DataFrame(results, columns=results.keys())
for result, idx in zip(results_raw, idxs):
    check_acc(result, idx)

df_frames_params.to_csv(os.path.join(
    LEGGED_GYM_ROOT_DIR,
    'vgcm/experiment_results/optimal_compensator_params.csv'))
