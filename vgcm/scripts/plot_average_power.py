import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from cycler import cycler

from vgcm.colours import colours
from vgcm.vgcm_ideal_model import COMPENSATION_OPTIONS
from legged_gym import LEGGED_GYM_ROOT_DIR

plt.rcParams['axes.prop_cycle'] = cycler(color=colours)


main_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'vgcm/experiment_results')
model = 'VGCM-cv'
path_to_files = os.path.join(main_dir, model)

dfs = {}
for file in os.listdir(path_to_files):
    filepath = os.path.join(path_to_files, file)
    for opt in COMPENSATION_OPTIONS:
        if opt in filepath:
            dfs[opt] = pd.read_csv(filepath)


WHEELS = [3, 7]
COMPENSATED = [0, 1, 2, 4, 5, 6]


def get_power_array(opt):
    df = dfs[opt]
    P = np.zeros(len(df))
    t = df["step"]
    for i in range(8):
        dq = df[f'dq{i}']
        full_tau = df[f'tau{i}']
        if opt != 'none' and i in COMPENSATED:
            idx = COMPENSATED.index(i)
            gc_tau = df[f'gc{idx}_tau']
            actuator_tau = full_tau - gc_tau
        else:
            actuator_tau = full_tau

        if i in WHEELS:
            Kt = 1. / (2*np.pi)
            R = 0.144
        else:
            Kt = 1. / (2*np.pi)
            R = 0.144

        P += actuator_tau * dq + np.square(actuator_tau) * R / np.square(Kt)
    return P, t


fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)

for ax, opt in zip(axes.flat, COMPENSATION_OPTIONS):
    P, t = get_power_array(opt)
    ax.plot(t, P, label=opt)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.legend()
    print(f"(Compensation: {opt}) Mean Power Draw {P.mean()}")

plt.show()
