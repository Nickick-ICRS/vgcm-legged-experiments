import onnxruntime as ort
import numpy as np
import os

from vgcm.robot_state import RobotState
from vgcm.vgcm_ideal_model import K_MAX, K_MIN, X_MAX, X_MIN


def load_onnx_model(onnx_path):
    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    print("ONNX model loaded successfully.")
    return session


class ControlSignal:
    def __init__(self, signal, controls_compensators=False, tk=None, tx=None):
        self.num_joints = len(signal)
        self.tau = signal
        self.controls_compensators = controls_compensators
        if self.controls_compensators:
            self.tk = tk
            self.tx = tx

    def __str__(self):
        output = f"Control Signal ({self.num_joints} joints)"
        output += f"Commanded Torques: {self.tau}"
        output += f"Controls Compensators: {self.controls_compensators}"
        return output


class Controller:
    def __init__(self, onnx_path, vgcm_alphas=np.zeros((2,))):
        self.session = load_onnx_model(onnx_path)
        self.controls_comps = True if 'vgcm' in os.path.basename(onnx_path) else False
        self.n_actions = 20 if self.controls_comps else 8
        self._prev_actions = np.zeros(self.n_actions, dtype=np.float32)
        self._prev_actions_scaled = np.zeros(self.n_actions, dtype=np.float32)
        self._last_ctrl = ControlSignal(np.zeros(8, dtype=np.float32), self.controls_comps)
        self._Kps = np.ones(8, dtype=np.float32) * 40.
        self._Kds = np.array([1.8, 1.8, 1.8, 0.5, 1.8, 1.8, 1.8, 0.5], dtype=np.float32)
        self._vgcm_alphas = vgcm_alphas

        self._scale_pos = 0.25
        self._scale_vel = 8

        self._obs_scale_base_lin_vel = 2.0
        self._obs_scale_base_ang_vel = 0.25
        self._obs_scale_g = 1. / 9.81
        self._obs_scale_pos = 1.
        self._obs_scale_vel = 0.05
        self._obs_scale_cmd = np.array([
            self._obs_scale_base_lin_vel, self._obs_scale_base_lin_vel, self._obs_scale_base_ang_vel])

        self._default_dof_pos = np.zeros(8, dtype=np.float32)
        self._torque_lims = np.array([80, 80, 80, 40, 80, 80, 80, 40], dtype=np.float32)
        if self.controls_comps:
            self._obs_scale_vgcm_k = np.array(2. / (K_MAX - K_MIN))
            self._obs_scale_vgcm_x = np.array(2. / (X_MAX - X_MIN))
            self._default_vgcm_k = (K_MIN + K_MAX) / 2
            self._default_vgcm_x = np.copy(self._default_dof_pos)[[0, 1, 2, 4, 5, 6]]
            self._vgcm_k_range = np.array(K_MAX - K_MIN)
            self._vgcm_x_range = np.array(X_MAX - X_MIN)

    def control(self, robot_state: RobotState, commands):
        state_input = self._extract_state(robot_state, commands)
        inputs = {self.session.get_inputs()[0].name: state_input}
        outputs = np.clip(self.session.run(None, inputs)[0], -100, 100)
        self._prev_actions = outputs
        WHEELS = [3, 7]
        actions_scaled = self._scale_pos * outputs[:8] + self._default_dof_pos
        actions_scaled[WHEELS] = self._scale_vel * outputs[WHEELS]
        self._prev_actions_scaled = actions_scaled
        # Wheels command v, others command x
        taus = self._Kps * (actions_scaled - robot_state.q) - self._Kps * robot_state.dq
        taus[WHEELS] = self._Kds[WHEELS] * (actions_scaled[WHEELS] - robot_state.dq[WHEELS])
        #taus = np.clip(taus, -self._torque_lims, self._torque_lims)

        if self.controls_comps:
            tk = np.clip(self._default_vgcm_k + outputs[8:14] * self._vgcm_k_range, K_MIN, K_MAX)
            tx = np.clip(self._default_vgcm_x + outputs[14:20] * self._vgcm_x_range, X_MIN, X_MAX)
            ctrl = ControlSignal(taus, True, tk, tx)
        else:
            ctrl = ControlSignal(taus, False)
        self._last_ctrl = ctrl
        return ctrl

    def _extract_state(self, state: RobotState, commands):
        pos_dofs = [0, 1, 2,
                    4, 5, 6]
        if self.controls_comps:
            obs = np.clip(np.concatenate([
                state.base_ang_vel * self._obs_scale_base_ang_vel,
                state.projected_gravity * self._obs_scale_g,
                (state.q - self._default_dof_pos)[pos_dofs] * self._obs_scale_pos,
                state.dq * self._obs_scale_vel,
                self._prev_actions,
                commands[:3] * self._obs_scale_cmd,
                # Forgot to scale when training
                self._vgcm_alphas,
                state.gc_k * self._obs_scale_vgcm_k,
                state.gc_x * self._obs_scale_vgcm_x,
            ]), -100, 100)
        else:
            obs = np.clip(np.concatenate([
                state.base_ang_vel * self._obs_scale_base_ang_vel,
                state.projected_gravity * self._obs_scale_g,
                (state.q - self._default_dof_pos)[pos_dofs] * self._obs_scale_pos,
                state.dq * self._obs_scale_vel,
                self._prev_actions,
                commands[:3] * self._obs_scale_cmd
            ]), -100, 100)
        return obs.astype(np.float32)
