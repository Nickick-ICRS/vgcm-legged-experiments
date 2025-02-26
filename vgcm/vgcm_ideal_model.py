from typing import Tuple, List

import torch


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

class VGCMTorsionSpringModel:
    def __init__(self, k_min: float, k_max: float, k_rate: float, preload_rate: float):
        self.k_min = k_min
        self.k_max = k_max
        self.k_rate = k_rate
        self.preload_rate = preload_rate

        self.preload = 0
        self.k = k_min

    def update(self, position, target_k, target_preload, dt):
        self.preload = self.update_param(self.preload, target_preload, self.preload_rate, dt)
        self.k = self.update_param(self.k, target_k, self.k_rate, dt)

        dx = position - self.preload
        f = self.k * dx
        return f

    def update_param(self, current, target, rate, dt):
        if current == target:
            return current
        delta = target - current
        max_delta = rate * dt
        if abs(delta) < abs(max_delta):
            return target
        delta = max_delta if delta > 0 else -max_delta
        return current + delta
