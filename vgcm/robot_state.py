import numpy as np

class RobotState:
    def __init__(self, has_gravity_compensators=False):
        self.num_joints = 8
        self.q = np.zeros(self.num_joints)
        self.dq = np.zeros(self.num_joints)
        self.tau = np.zeros(self.num_joints)

        self.stamp = 0
        self.imu_data_quat = np.zeros(4)
        self.imu_data_gyro = np.zeros(3)
        self.imu_data_acc = np.zeros(3)

        self.has_gravity_compensators = has_gravity_compensators
        if self.has_gravity_compensators:
            # Don't compensate wheels
            self.num_compensators = num_joints-2
            # TODO: Check what these parameters need to be
            self.gc_a = np.zeros(self.num_compensators)
            self.gc_b = np.zeros(self.num_compensators)
            # Set Point
            self.gc_sp = np.zeros(self.num_compensators)
            # Current Position
            self.gc_cp = np.zeros(self.num_compensators)
        else:
            self.num_compensators = 0

    def __str__(self):
        state_str = f"RobotState (stamp={self.stamp}, num_joints={self.num_joints}):\n"
        state_str += f"  Joint Positions (q): {self.q}\n"
        state_str += f"  Joint Velocities (dq): {self.dq}\n"
        state_str += f"  Joint Torques (tau): {self.tau}\n"
        state_str += f"  IMU Data (Quaternion): {self.imu_data_quat}\n"
        state_str += f"  IMU Data (Gyroscope): {self.imu_data_gyro}\n"
        state_str += f"  Has Gravity Compensators: {self.has_gravity_compensators}\n"

        if self.has_gravity_compensators:
            state_str += f"  Number of Gravity Compensators: {self.num_compensators}\n"
            state_str += f"  Gravity Compensator A: {self.gc_a}\n"
            state_str += f"  Gravity Compensator B: {self.gc_b}\n"
            state_str += f"  Gravity Compensator Set Point: {self.gc_sp}\n"
            state_str += f"  Gravity Compensator Current Position: {self.gc_cp}\n"
        
        return state_str