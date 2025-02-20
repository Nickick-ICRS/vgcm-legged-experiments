from typing import Tuple

import numpy as np


class VGCMParameters:
    def __init__(self, alpha: float = 100.0, lmbda: float = 0.0,
                 tau_limits: Tuple[float, float] = (0., 10.),
                 theta_limits: Tuple[float, float] = (-3.141, 3.141),
                 moment_arm: float = 0.3, axis: np.ndarray = np.array([0, 1, 0])):
        """
        @param alpha The adjustment rate (Nm/s)
        @param lmbda Input lag (s)
        @param tau_limits (minimum, maximum) Torque limits (Nm)
        @param theta_limits (minimum, maximum) Position limits (rads)
        @param moment_arm Length of the moment arm (m)
        """
        self.alpha = alpha
        self.lmbda = lmbda
        self.tau_lims = tau_limits
        self.theta_lims = theta_limits
        self.moment_arm = moment_arm
        self.axis = axis


class VGCMIdealModel(object):
    def __init__(self, params: VGCMParameters = VGCMParameters()):
        self.params = params
        self.command_time = 0
        self.command_pos = 0
        self.command_torque = 0
        self._r0 = np.array([0, 0, -1])
        self._kxr = np.cross(self.params.axis, self._r0)
        self._kdr = np.dot(self.params.axis, self._r0)
    
    def update(self, position: float, current_time: float) -> float:
        """
        @detail Update the VGCM and calculate applied force
        @param position Current joint position
        @param current_time Current simulation time

        @returns Torque applied by VGCMIdealModel
        """

    def command(self, position: float, torque: float, current_time: float):
        """
        @detail Command the VGCM to apply a given torque at a given position

        @param position The position to apply the torque at
        @param torque The torque to apply
        @param current_time Current simulation time
        """
        self.command_pos = position
        self.command_torque = torque
        # Don't apply input lag if already moving
        if current_time >= self.command_time + self.params.lmbda:
            self.command_time = current_time - self.params.lmbda
        else:
            self.command_time = current_time

    def _calculate_torque(self, position, current_time):
        raise NotImplementedError()

    def _update_internal_model(self, current_time):
        """
        @detail Internal model used by inherited classes
        """
        raise NotImplementedError()

    def calculate_expected_torque_to_compensate(self, position, mass, gravity, ext_force):
        c = np.cos(position)[:, np.newaxis]
        s = np.sin(position)[:, np.newaxis]
        r = c * self._r0 + s * self._kxr + (1-c) * self._kdr * self.params.axis
        F = -mass * gravity + ext_force
        tau = np.cross(r, F)
        return tau.dot(self.params.axis)
