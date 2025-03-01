import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

from legged_gym import LEGGED_GYM_ROOT_DIR
from vgcm.colours import colours
plt.rcParams['axes.prop_cycle'] = cycler(color=colours)


alpha = 2.
T = 10.
offset = -T / 4


def lissajous(t):
    return np.array([
        alpha * np.cos(2 * np.pi * (t+offset) / T),
        alpha * np.sin(4 * np.pi * (t+offset) / T) / 2])


t = np.linspace(0, T, 50)


data_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'vgcm/experiment_results')
# df_cv_non = pd.read_csv(os.path.join(data_dir, 'VGCM-cv/lissajous_trajectory_experiment_none_compensation_results_0.csv'))
df_cv_low = pd.read_csv(os.path.join(data_dir, 'VGCM-cv/lissajous_trajectory_experiment_low_compensation_results_0.csv'))
# df_cv_med = pd.read_csv(os.path.join(data_dir, 'VGCM-cv/lissajous_trajectory_experiment_medium_compensation_results_0.csv'))
# df_cv_hig = pd.read_csv(os.path.join(data_dir, 'VGCM-cv/lissajous_trajectory_experiment_high_compensation_results_0.csv'))
# df_ma_non = pd.read_csv(os.path.join(data_dir, 'VGCM-ma/lissajous_trajectory_experiment_none_compensation_results_0.csv'))
df_ma_low = pd.read_csv(os.path.join(data_dir, 'VGCM-ma/lissajous_trajectory_experiment_low_compensation_results_0.csv'))
# df_ma_med = pd.read_csv(os.path.join(data_dir, 'VGCM-ma/lissajous_trajectory_experiment_medium_compensation_results_0.csv'))
# df_ma_hig = pd.read_csv(os.path.join(data_dir, 'VGCM-ma/lissajous_trajectory_experiment_high_compensation_results_0.csv'))
# df_dc_non = pd.read_csv(os.path.join(data_dir, 'VGCM-dc/lissajous_trajectory_experiment_none_compensation_results_0.csv'))
# df_dc_low = pd.read_csv(os.path.join(data_dir, 'VGCM-dc/lissajous_trajectory_experiment_low_compensation_results_0.csv'))
# df_dc_med = pd.read_csv(os.path.join(data_dir, 'VGCM-dc/lissajous_trajectory_experiment_medium_compensation_results_0.csv'))
# df_dc_hig = pd.read_csv(os.path.join(data_dir, 'VGCM-dc/lissajous_trajectory_experiment_high_compensation_results_0.csv'))


fig, ax = plt.subplots()
liss = lissajous(t)
ax.plot(liss[0], liss[1], label="Requested")
# ax.plot(df_cv_non['base_pos_x'], df_cv_non['base_pos_y'], label="VGCM-cv-none")
ax.plot(df_cv_low['base_pos_x'], df_cv_low['base_pos_y'], label="VGCM-cv-low")
# ax.plot(df_cv_med['base_pos_x'], df_cv_med['base_pos_y'], label="VGCM-cv-medium")
# ax.plot(df_cv_hig['base_pos_x'], df_cv_hig['base_pos_y'], label="VGCM-cv-high")
# ax.plot(df_ma_non['base_pos_x'], df_ma_non['base_pos_y'], label="VGCM-ma-none")
ax.plot(df_ma_low['base_pos_x'], df_ma_low['base_pos_y'], label="VGCM-ma-low")
# ax.plot(df_ma_med['base_pos_x'], df_ma_med['base_pos_y'], label="VGCM-ma-medium")
# ax.plot(df_ma_hig['base_pos_x'], df_ma_hig['base_pos_y'], label="VGCM-ma-high")
# ax.plot(df_dc_non['base_pos_x'], df_dc_non['base_pos_y'], label="VGCM-dc-none")
# ax.plot(df_dc_low['base_pos_x'], df_dc_low['base_pos_y'], label="VGCM-dc-low")
# ax.plot(df_dc_med['base_pos_x'], df_dc_med['base_pos_y'], label="VGCM-dc-medium")
# ax.plot(df_dc_hig['base_pos_x'], df_dc_hig['base_pos_y'], label="VGCM-dc-high")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.legend()
fig.set_tight_layout(True)
plt.show()
