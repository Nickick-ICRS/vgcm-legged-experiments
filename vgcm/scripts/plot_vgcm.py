from vgcm.vgcm_ideal_model import VGCMIdealModel, VGCMParameters

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cycler import cycler
import numpy as np
from scipy.spatial.transform import Rotation


colours = [
    '#4466aa', '#66ccee', '#228833', '#ccbb44', '#ee6677', '#aa3377', '#bbbbbb'
]
plt.rcParams['axes.prop_cycle'] = cycler(color=colours)


params = VGCMParameters(
    alpha=1e9, lmbda=0, tau_limits=(0, 10), theta_limits=(0, np.pi),
    moment_arm=1., axis=np.array([0, 1, 0]))
model = VGCMIdealModel(params)

theta = np.linspace(0, np.pi, 50, True)

g = np.array([0, 0, -9.81])
ext_f = np.zeros_like(g)
force_10kg = model.calculate_expected_torque_to_compensate(theta, 10, g, ext_f)
force_20kg = model.calculate_expected_torque_to_compensate(theta, 20, g, ext_f)
force_30kg = model.calculate_expected_torque_to_compensate(theta, 30, g, ext_f)

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
        print(rot)
        R = Rotation.from_rotvec(axis * rot)
        gr = R.as_matrix() @ g
        line_10kg.set_ydata(model.calculate_expected_torque_to_compensate(theta, 10, gr, ext_f))
        line_20kg.set_ydata(model.calculate_expected_torque_to_compensate(theta, 20, gr, ext_f))
        line_30kg.set_ydata(model.calculate_expected_torque_to_compensate(theta, 30, gr, ext_f))
        ax.set_title(f"Moment Induced by Gravity for a Revolute Joint with Unit Length Arm\n"
                     f"With Relative Gravity {gr}")
        ax.legend()
        return [line_10kg, line_20kg, line_30kg]
    return update

np.set_printoptions(precision=2)

n_frames = 100
axis = np.array([0.707, 0.707, 0])
ani_10kg = animation.FuncAnimation(fig, make_update(n_frames, axis),
                                   frames=n_frames, interval=50, blit=False)
plt.show()
