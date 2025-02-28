from typing import Tuple, List

import os
import torch
import numpy as np
import pandas as pd
from legged_gym import LEGGED_GYM_ROOT_DIR


class VGCMParameters(torch.nn.Module):
    def __init__(self, alpha: float = 100.0, lmbda: float = 0.0,
                 tau_limits: Tuple[float, float] = (0., 10.),
                 theta_limits: Tuple[float, float] = (-3.141, 3.141),
                 axis: torch.Tensor = torch.tensor([0, 1, 0], dtype=torch.float32)):
        """
        @param alpha The adjustment rate (Nm/s)
        @param lmbda Input lag (s)
        @param tau_limits (minimum, maximum) Torque limits (Nm)
        @param theta_limits (minimum, maximum) Position limits (rads)
        @param axis Axis of rotation of the joint
        """
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.float32), requires_grad=False)
        self.lmbda = torch.nn.Parameter(torch.tensor([lmbda], dtype=torch.float32), requires_grad=False)
        self.tau_lims = torch.nn.Parameter(torch.tensor(tau_limits, dtype=torch.float32), requires_grad=False)
        self.theta_lims = torch.nn.Parameter(torch.tensor(theta_limits, dtype=torch.float32), requires_grad=False)
        self.axis = torch.nn.Parameter(axis, requires_grad=False)


class VGCMIdealModel(torch.nn.Module):
    def __init__(self, params: List[VGCMParameters] = [VGCMParameters()]):
        super().__init__()
        self.alphas = torch.stack([p.alpha for p in params])
        self.lmdas = torch.stack([p.lmbda for p in params])
        self.tau_lims = torch.stack([p.tau_lims for p in params])
        self.theta_lims = torch.stack([p.theta_lims for p in params])
        self.axes = torch.stack([p.axis for p in params])
        self._r0 = torch.stack([torch.tensor([0, 0, -1], dtype=torch.float32) for p in params])
        self._kxr = torch.cross(self.axes, self._r0)
        self._kdr = (self.axes * self._r0).sum(-1, keepdims=True)

    def calculate_expected_torque_to_compensate(self, positions, masses, gravities, ext_forces):
        c = torch.cos(positions)
        s = torch.sin(positions)
        r = c * self._r0 + s * self._kxr + (1-c) * self._kdr * self.axes
        F = -masses * gravities + ext_forces
        tau = torch.cross(r, F)
        return (tau * self.axes).sum(-1, keepdims=True)

    def calculate_linear_model(self, positions, target_positions, mass, gravity, ext_force):
        dx = torch.ones_like(target_positions) * 0.01
        grad_positions = torch.stack([target_positions-dx, target_positions, target_positions+dx], dim=-1).squeeze(-2)
        req_torque_0 = self.calculate_expected_torque_to_compensate(
            grad_positions[:, 0].unsqueeze(-1), mass, gravity, ext_force)
        req_torque_1 = self.calculate_expected_torque_to_compensate(
            grad_positions[:, 1].unsqueeze(-1), mass, gravity, ext_force)
        req_torque_2 = self.calculate_expected_torque_to_compensate(
            grad_positions[:, 2].unsqueeze(-1), mass, gravity, ext_force)
        m = (req_torque_2 - req_torque_0) / (2 * dx)
        c = req_torque_1
        return m * (positions - target_positions) + c, m, c


class VGCMSpringModel:
    def __init__(self, k_min: float, k_max: float, x_min: float, x_max: float,
                 k_rate: float, preload_rate: float):
        self.k_min = k_min
        self.k_max = k_max
        self.x_min = x_min
        self.x_max = x_max
        self.k_rate = k_rate
        self.preload_rate = preload_rate

        # This is actually the preload position not the preload force
        self.preload = (x_min + x_max) / 2
        self.k = (k_min + k_max) / 2

        self.tk = self.k
        self.tx = self.preload
        self.EPSILON = 1e-2

    def update(self, position, target_k, target_preload, dt):
        self.preload = self.update_params(
            self.preload, target_preload, self.preload_rate, dt)
        self.k = self.update_params(self.k, target_k, self.k_rate, dt)

        dx = position - self.preload
        f = self.k * dx
        self.tk = target_k
        self.tx = target_preload
        return f

    def update_params(self, current, target, rate, dt):
        if current == target:
            return current
        delta = target - current
        max_delta = rate * dt
        if abs(delta) < abs(max_delta):
            return target
        delta = max_delta if delta > 0 else -max_delta
        return current + delta

    def calc_k_x(self, r0, axis, tau, gdir, theta):
        # Estimate gravity force that would result in tau
        rxd = np.cross(r0, gdir)
        rxddk = np.dot(rxd, axis)
        f = (tau / rxddk) * gdir

        def get_tau(theta):
            c = np.cos(theta)
            s = np.sin(theta)
            r = c * r0 + s * np.cross(axis, r0) + (1-c) * np.dot(axis, r0) * axis
            return np.dot(np.cross(r, f), axis)

        delta = 0.01
        t0 = get_tau(-delta)
        t1 = get_tau(delta)
        k = np.clip(np.abs((t1 - t0) / (2 * delta)), self.k_min, self.k_max)
        k = np.abs(t1 - t0) / (2 * delta)
        sign = 1 if t1 > t0 else -1
        self.sign = sign
        dx = sign * tau / k
        x = np.clip(theta - dx, self.x_min, self.x_max)

        return k, x


COMPENSATION_OPTIONS = ['none', 'low', 'medium', 'high']

# From find_optimal_compensator_parameters.py
K_MIN = [29.12, 727.19, 727.25, 178.57, 58.68, 281.43]
K_MAX = [33.13, 735.13, 756.52, 180.71, 58.97, 281.95]
X_MIN = [-0.1623, -0.8495, -1.1895, -0.1413, -2.8678, -2.018]
X_MAX = [1.9407, 0.0380, 0.6501, 1.2907, 0.3654, 1.6530]

params_file = os.path.join(
    LEGGED_GYM_ROOT_DIR,
    'vgcm/experiment_results/optimal_compensator_params.csv')

if os.path.exists(params_file):
    df = pd.read_csv(params_file)
    K_MIN = df.filter(like="k_min").to_numpy().squeeze()
    K_MAX = df.filter(like="k_max").to_numpy().squeeze()
    X_MIN = df.filter(like="x_min").to_numpy().squeeze()
    X_MAX = df.filter(like="x_max").to_numpy().squeeze()
else:
    print("Warning: Compensator params have not yet been generated. Please run `find_optimal_compensator_parameters'")

# Seconds to get from min to max
ALPHA_MUL_K = {
    'low': 20, 'medium': 10, 'high': 5,
}
ALPHA_MUL_X = {
    'low': 20, 'medium': 10, 'high': 5,
}


def make_compensator(fuzzy_adjustment_speed: str, joint_id: int):
    f"""
    fuzzy_adjustment_speed: one of {COMPENSATION_OPTIONS}, picks adjustment rates
    joint_id: 0, 1, 2, 3, 4, 5, does not include wheels, abad_L -> knee_R
    """
    if fuzzy_adjustment_speed not in COMPENSATION_OPTIONS:
        assert f"{fuzzy_adjustment_speed} not one of {COMPENSATION_OPTIONS}"
    if fuzzy_adjustment_speed == 'none':
        return None
    else:
        k_min = K_MIN[joint_id]
        k_max = K_MAX[joint_id]
        x_min = X_MIN[joint_id]
        x_max = X_MAX[joint_id]
        alpha_k = (k_max - k_min) / ALPHA_MUL_K[fuzzy_adjustment_speed]
        alpha_x = (x_max - x_min) / ALPHA_MUL_X[fuzzy_adjustment_speed]
        alpha_k = 9e9
        alpha_x = 9e9
        return VGCMSpringModel(k_min, k_max, x_min, x_max, alpha_k, alpha_x)
