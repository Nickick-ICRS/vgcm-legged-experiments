import argparse
import os

from legged_gym import LEGGED_GYM_ROOT_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycler import cycler
from vgcm.colours import colours
plt.rcParams['axes.prop_cycle'] = cycler(color=colours)


results_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'vgcm/experiment_results')
filepath = os.path.join(results_dir, 'basic_linear_experiment_results_0.csv')
filepath = os.path.join(results_dir, 'lissajous_trajectory_experiment_results_0.csv')


df = pd.read_csv(filepath)

joint_names = ["abad_L_Joint", "hip_L_Joint", "knee_L_Joint",# "wheel_L_Joint",
               "abad_R_Joint", "hip_R_Joint", "knee_R_Joint",# "wheel_R_Joint"
]
joint_ids = [0, 1, 2, 4, 5, 6]

ax = df[[f"tau{i}" for i in joint_ids]].plot(kind="box", figsize=(10, 6))
plt.xticks(ticks=range(1, 7), labels=joint_names)
plt.title("Joint Torque Distributions")
plt.ylabel("Torque (Nm)")
plt.show()

ax = df[[f"q{i}" for i in joint_ids]].plot(kind="box", figsize=(10, 6))
plt.xticks(ticks=range(1, 7), labels=joint_names)
plt.title("Joint Position Distributions")
plt.ylabel("Position (rad)")
plt.show()

fig, ax = plt.subplots()
ax.plot(df["q0"], df["tau0"])
ax.set_xlabel("Position (rad)")
ax.set_ylabel("Torque (Nm)")
ax.set_title(f"Position and torque for {joint_names[0]}")
fig.set_tight_layout(True)
plt.show()