import numpy as np

class RobotState:
    def __init__(self, has_gravity_compensators=False):
        self.num_joints = 8
        self.q = np.zeros(self.num_joints)
        self.dq = np.zeros(self.num_joints)
        self.tau = np.zeros(self.num_joints)

        self.base_quat = np.zeros(4)
        self.base_pos = np.zeros(3)
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.projected_gravity = np.zeros(3)

        self.stamp = 0
        self.imu_data_quat = np.zeros(4)
        self.imu_data_gyro = np.zeros(3)
        self.imu_data_acc = np.zeros(3)

        self.has_gravity_compensators = has_gravity_compensators
        if self.has_gravity_compensators:
            # Don't compensate wheels
            self.num_compensators = num_joints-2
            # Spring constant
            self.gc_k = np.zeros(self.num_compensators)
            # Zero position
            self.gc_x = np.zeros(self.num_compensators)
            # Target spring constant 
            self.gc_tk = np.zeros(self.num_compensators)
            # Target zero position
            self.gc_tx = np.zeros(self.num_compensators)
        else:
            self.num_compensators = 0

    def to_dict(self):
        entry = {
            "step": self.stamp,
            **{"q"+str(i): q for i, q in enumerate(self.q)},
            **{"dq"+str(i): dq for i, dq in enumerate(self.dq)},
            **{"tau"+str(i): tau for i, tau in enumerate(self.tau)},
            "base_pos_x": self.base_pos[0],
            "base_pos_y": self.base_pos[1],
            "base_pos_z": self.base_pos[2],
            "base_quat_x": self.base_quat[0],
            "base_quat_y": self.base_quat[1],
            "base_quat_z": self.base_quat[2],
            "base_quat_w": self.base_quat[3],
            "base_lin_vel_x": self.base_lin_vel[0],
            "base_lin_vel_y": self.base_lin_vel[1],
            "base_lin_vel_z": self.base_lin_vel[2],
            "base_ang_vel_x": self.base_ang_vel[0],
            "base_ang_vel_y": self.base_ang_vel[1],
            "base_ang_vel_z": self.base_ang_vel[2],
        }
        if self.has_gravity_compensators:
            for i in range(self.num_compensators):
                entry[f"gc{i}_stiffness"] = self.gc_k[i]
                entry[f"gc{i}_zero_pos"] = self.gc_x[i]
                entry[f"gc{i}_target_stiffness"] = self.gc_tk[i]
                entry[f"gc{i}_target_zero_pos"] = self.gc_tx[i]
        return entry

    def __str__(self):
        state_str = f"RobotState (stamp={self.stamp}, num_joints={self.num_joints}):\n"
        state_str += f"  Joint Positions (q): {self.q}\n"
        state_str += f"  Joint Velocities (dq): {self.dq}\n"
        state_str += f"  Joint Torques (tau): {self.tau}\n"

        state_str += f"  Base Pos: {self.base_pos}\n"
        state_str += f"  Base Quat: {self.base_quat}\n"
        state_str += f"  Base Lin Vel: {self.base_lin_vel}\n"
        state_str += f"  Base Ang Vel: {self.base_ang_vel}\n"
        state_str += f"  Projected Gravity: {self.projected_gravity}\n"

        state_str += f"  IMU Data (Quaternion): {self.imu_data_quat}\n"
        state_str += f"  IMU Data (Gyroscope): {self.imu_data_gyro}\n"
        state_str += f"  IMU Data (Accelerometer): {self.imu_data_acc}\n"
        state_str += f"  Has Gravity Compensators: {self.has_gravity_compensators}\n"

        if self.has_gravity_compensators:
            state_str += f"  Number of Gravity Compensators: {self.num_compensators}\n"
            state_str += f"  Gravity Compensator A: {self.gc_a}\n"
            state_str += f"  Gravity Compensator B: {self.gc_b}\n"
            state_str += f"  Gravity Compensator Set Point: {self.gc_sp}\n"
            state_str += f"  Gravity Compensator Current Position: {self.gc_cp}\n"
        
        return state_str