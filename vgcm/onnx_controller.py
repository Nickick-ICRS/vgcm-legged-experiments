import onnxruntime as ort
import numpy as np

from vgcm.robot_state import RobotState


def load_onnx_model(onnx_path):
    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    print("ONNX model loaded successfully.")
    return session


class ControlSignal:
    def __init__(self, signal, controls_compensators=False):
        self.num_joints = len(signal)
        self.tau = signal
        self.controls_compensators = controls_compensators

    def __str__(self):
        output = f"Control Signal ({self.num_joints} joints)"
        output += f"Commanded Torques: {self.tau}"
        output += f"Controls Compensators: {self.controls_compensators}"
        return output


class Controller:
    def __init__(self, onnx_path):
        self.session = load_onnx_model(onnx_path)
        self._prev_actions = np.zeros(8, dtype=np.float32)
        self._last_ctrl = ControlSignal(np.zeros(8, dtype=np.float32))
        self._Kps = np.ones(8, dtype=np.float32) * 40.
        self._Kds = np.array([1.8, 1.8, 1.8, 0.5, 1.8, 1.8, 1.8, 0.5], dtype=np.float32)

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

    def control(self, robot_state: RobotState, commands):
        state_input = self._extract_state(robot_state, commands)
        inputs = {self.session.get_inputs()[0].name: state_input}
        outputs = self.session.run(None, inputs)[0]
        self._prev_actions = outputs
        # Wheels command v, others command x
        taus = self._Kps * (self._scale_pos * outputs + self._default_dof_pos - robot_state.q) - self._Kps * robot_state.dq
        WHEELS = [3, 7]
        taus[WHEELS] = self._Kds[WHEELS] * (self._scale_vel * outputs[WHEELS] - robot_state.dq[WHEELS])
        taus = np.clip(taus, -self._torque_lims, self._torque_lims)

        ctrl = ControlSignal(taus, self._last_ctrl.controls_compensators)
        self._last_ctrl = ctrl
        return ctrl

    def _extract_state(self, state: RobotState, commands):
        pos_dofs = [0, 1, 2,
                    4, 5, 6]
        obs = np.concatenate([
            state.base_ang_vel * self._obs_scale_base_ang_vel,
            state.projected_gravity * self._obs_scale_g,
            (state.q - self._default_dof_pos)[pos_dofs] * self._obs_scale_pos,
            state.dq * self._obs_scale_vel,
            self._prev_actions,
            commands[:3] * self._obs_scale_cmd
        ])
        return obs.astype(np.float32)
