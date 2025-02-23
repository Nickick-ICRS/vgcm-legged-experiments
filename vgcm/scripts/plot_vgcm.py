from vgcm.vgcm_ideal_model import VGCMIdealModel, VGCMParameters

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cycler import cycler
import numpy as np
import torch
from scipy.spatial.transform import Rotation


colours = [
    '#4466aa', '#66ccee', '#228833', '#ccbb44', '#ee6677', '#aa3377', '#bbbbbb'
]
plt.rcParams['axes.prop_cycle'] = cycler(color=colours)


params = VGCMParameters(
    alpha=1e9, lmbda=0, tau_limits=(0, 10), theta_limits=(np.pi/8, np.pi-np.pi/8),
    axis=torch.tensor([0, 1, 0], dtype=torch.float32))
model = VGCMIdealModel([params])

theta = torch.linspace(params.theta_lims[0], params.theta_lims[1], 50).unsqueeze(-1)

g = torch.stack([torch.tensor([0, 0, -9.81], dtype=torch.float32) for i in range(theta.shape[0])])
ext_f = torch.zeros_like(g)
m10 = torch.tensor([10 for i in range(theta.shape[0])], dtype=torch.float32).unsqueeze(-1)
m20 = torch.tensor([20 for i in range(theta.shape[0])], dtype=torch.float32).unsqueeze(-1)
m30 = torch.tensor([30 for i in range(theta.shape[0])], dtype=torch.float32).unsqueeze(-1)
force_10kg = model.calculate_expected_torque_to_compensate(theta, m10, g, ext_f)
force_20kg = model.calculate_expected_torque_to_compensate(theta, m20, g, ext_f)
force_30kg = model.calculate_expected_torque_to_compensate(theta, m30, g, ext_f)

fig, ax = plt.subplots()
ax.plot(theta, force_10kg, label="Mass: 10 Kg")
ax.plot(theta, force_20kg, label="Mass: 20 Kg")
ax.plot(theta, force_30kg, label="Mass: 30 Kg")
ax.set_xlabel("Arm Angle (rads)")
ax.set_ylabel("Joint Torque (Nm)")
ax.set_title("Moment Induced by Gravity for a Revolute Joint with Unit Length Arm")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_xlim(0, np.pi)
ax.set_ylim(-300, 300)
ax.set_xlabel("Arm Angle (rads)")
ax.set_ylabel("Joint Torque (Nm)")
ax.set_title("Moment Induced by Gravity for a Revolute Joint with Unit Length Arm")
ax.legend()

line_10kg, = ax.plot(theta, force_10kg, label="Mass: 10 Kg")
line_20kg, = ax.plot(theta, force_20kg, label="Mass: 20 Kg")
line_30kg, = ax.plot(theta, force_30kg, label="Mass: 30 Kg")

def make_update(n_frames, axis):
    def update(frame):
        rot = 2 * np.pi * frame / n_frames
        R = torch.stack([torch.tensor(Rotation.from_rotvec(axis * rot).as_matrix(), dtype=torch.float32) for i in range(theta.shape[0])])
        gr = torch.bmm(R, g.unsqueeze(-1)).squeeze(-1)
        line_10kg.set_ydata(model.calculate_expected_torque_to_compensate(theta, m10, gr, ext_f))
        line_20kg.set_ydata(model.calculate_expected_torque_to_compensate(theta, m20, gr, ext_f))
        line_30kg.set_ydata(model.calculate_expected_torque_to_compensate(theta, m30, gr, ext_f))
        ax.set_title(f"Moment Induced by Gravity for a Revolute Joint with Unit Length Arm\n"
                     f"With Relative Gravity {gr[0]}")
        ax.legend()
        return [line_10kg, line_20kg, line_30kg]
    return update

n_frames = 100
axis = np.array([0.707, 0.707, 0])
ani = animation.FuncAnimation(fig, make_update(n_frames, axis),
                              frames=n_frames, interval=50, blit=False)
plt.show()

fig, ax = plt.subplots()
ax.set_xlim(0, np.pi)
ax.set_ylim(-200, 200)
ax.set_xlabel("Arm Angle (rads)")
ax.set_ylabel("Joint Torque (Nm)")
ax.set_title("Moment Induced by Gravity for a Revolute Joint with Unit Length Arm")
ax.legend()

line_20kg, = ax.plot(theta, force_10kg, label="Required Force")
linear_20kg, = ax.plot(theta, force_10kg, label="Compensation")
dot, = ax.plot(0, 0, 'ro', label="Linearisaion Point")

def make_update(n_frames, axis):
    def update(frame):
        rot = np.pi * np.cos(2 * np.pi * frame / n_frames) / 4
        R = torch.stack([torch.tensor(Rotation.from_rotvec(axis * rot).as_matrix(), dtype=torch.float32) for i in range(theta.shape[0])])
        gr = torch.bmm(R, g.unsqueeze(-1)).squeeze(-1)

        target_thetas = torch.tensor([0.8 + 0.2 * np.sin(4*np.pi*frame / n_frames) for i in range(theta.shape[0])], dtype=torch.float32).unsqueeze(-1)

        line_20kg.set_ydata(model.calculate_expected_torque_to_compensate(theta, m20, gr, ext_f))
        linear_20kg.set_ydata(model.calculate_linear_model(theta, target_thetas, m20, gr, ext_f))
        dot.set_xdata([target_thetas[0]])
        target_forces = model.calculate_expected_torque_to_compensate(target_thetas, m20, gr, ext_f)
        dot.set_ydata([target_forces[0]])
        ax.legend()
        return [line_20kg, linear_20kg, dot]
    return update

n_frames=200
axis = np.array([1, 0, 0])
ani = animation.FuncAnimation(fig, make_update(n_frames, axis),
                              frames=n_frames, interval=50, blit=False)
plt.show()