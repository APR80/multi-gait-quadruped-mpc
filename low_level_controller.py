import numpy as np
from parameters import Params


class LowLevelController:
    """
    torque controller vectorized for speed.
    """

    def __init__(self, robot_data, gait_generator, swing_foot_trajectory_generator):
        self.gait_generator = gait_generator
        self.swing_foot_trajectory_generator = swing_foot_trajectory_generator
        self.robot_data = robot_data
        self.Kp = Params.Kp_swing
        self.Kd = Params.Kd_swing
        self.stance_damping = np.zeros(12)
        self.compensate_stance_bias = False

        self.torque_cmds = np.zeros(12, dtype=np.float32)

    def update(self, data, contact_forces):
        is_swing = np.asarray(self.gait_generator.get_swing_contact(), dtype=bool)
        f_stance = -contact_forces.reshape(4, 3)
        pos_err_base = (
            self.swing_foot_trajectory_generator.pos_des_base
            - self.robot_data.pos_feet_rel_base
        )
        vel_err_base = (
            self.swing_foot_trajectory_generator.vel_des_base
            - self.robot_data.vel_foot_rel_base
        )
        # Rotate Errors to World Frame
        R_base = self.robot_data.R_base
        pos_err_world = pos_err_base @ R_base.T
        vel_err_world = vel_err_base @ R_base.T

        # F = (Kp @ err_pos.T).T + (Kd @ err_vel.T).T
        f_swing = (self.Kp @ pos_err_world.T).T + (self.Kd @ vel_err_world.T).T

        F_combined = np.where(is_swing[:, None], f_swing, f_stance)

        J_blocks = np.stack(
            [
                J[:, 6 + 3 * i : 6 + 3 * (i + 1)]
                for i, J in enumerate(self.robot_data.Jv_feet)
            ]
        )  # Result shape: (4, 3, 3) -> (Leg, XYZ, Joint)

        swing_bias = (data.qfrc_bias[6:] - data.qfrc_passive[6:]).reshape(4, 3)
        stand_bias = data.qfrc_bias[6:].reshape(4, 3)
        feedforward = np.where(
            is_swing[:, None],
            swing_bias,
            np.where(self.compensate_stance_bias, stand_bias, 0),
        ).flatten()
        self.torque_cmds = (
            np.einsum("lji, lj -> li", J_blocks, F_combined).flatten() + feedforward
        )
        self.torque_cmds -= (
            np.repeat(~is_swing, 3) * self.stance_damping * data.qvel[6:]
        )

        return self.torque_cmds
