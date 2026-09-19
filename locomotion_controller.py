import mujoco as mj
import numpy as np
from parameters import Params
from robot_data import RobotData
from gait_generator import GaitGenerator
from gait_generator import GaitType
from mpc import ConvexMPC
from swing_foot_trajectory_generator import SwingFootTrajectoryGenerator
from low_level_controller import LowLevelController


class LocomotionController:
    def __init__(
        self,
        model,
        data,
        gait=GaitType.TROT,
        front_clearance=None,
        rear_clearance=None,
    ):
        self.model = model
        self.data = data
        self.robot_data = RobotData(model, data)
        self._reset()
        Params.dt_control = model.opt.timestep
        Params.iteration_between_mpc = max(1, round(0.02 / Params.dt_control))
        Params.dt_mpc = Params.dt_control * Params.iteration_between_mpc
        Params.init_from_model(model, data=data)
        self.gait = GaitGenerator(gait_type=gait)
        self.mpc = ConvexMPC(self.robot_data)
        self.swing = SwingFootTrajectoryGenerator(self.robot_data, self.gait)
        self.front_clearance = front_clearance
        self.rear_clearance = rear_clearance
        self.mpc.swing = self.swing
        self.low_level = LowLevelController(self.robot_data, self.gait, self.swing)
        self._configure_gait()

        self.v_des_current = np.zeros(3)
        self.v_des_target = np.zeros(3)
        self.yaw_rate_des_current = 0.0
        self.yaw_rate_des_target = 0.0
        self.alpha = 1.0 - np.exp(-Params.dt_control / 0.2)
        self.iter_counter = 0

    def _reset(self):
        mj.mj_resetData(self.model, self.data)
        q_pos_init = np.array(
            [
                0,
                0,
                Params.base_height_des,
                1,
                0,
                0,
                0,  # Quaternion (w, x, y, z)
                0,
                0.8,
                -1.6,  # FL
                0,
                0.8,
                -1.6,  # FR
                0,
                0.8,
                -1.6,  # RL
                0,
                0.8,
                -1.6,  # RR
            ]
        )
        self.data.qpos[:] = q_pos_init
        mj.mj_forward(self.model, self.data)
        clearance = (
            self.data.site_xpos[self.robot_data.foot_geom_ids, 2]
            - self.robot_data.foot_radii
        )
        self.data.qpos[2] += 0.001 - np.min(clearance)
        mj.mj_forward(self.model, self.data)

    def _filter_commands(self):
        """smooth transition from current speed to target speed."""
        target = np.asarray(self.v_des_target, dtype=float).copy()
        speed = np.linalg.norm(target[:2])
        if speed > self.gait.max_speed:
            target *= self.gait.max_speed / speed
        yaw_target = self.yaw_rate_des_target
        if self.gait.transitioning or self.gait.gait_type == GaitType.STAND:
            target = np.zeros(3)
            yaw_target = 0.0
        self.v_des_current = (1 - self.alpha) * self.v_des_current + self.alpha * target

        self.yaw_rate_des_current = (
            1 - self.alpha
        ) * self.yaw_rate_des_current + self.alpha * yaw_target

    def request_gait(self, gait):
        self.gait.request_gait(gait)
        self.mpc.last_contact = None
        self.mpc.roll_init = self.mpc.pitch_init = 0.0

    def _configure_gait(self):
        self._active_gait = self.gait.gait_type
        bounding = self._active_gait == GaitType.BOUNDING
        # A bound pitches the body, so the front feet need more room than the rear.
        front = (
            (Params.bound_front_clearance if bounding else Params.swing_height)
            if self.front_clearance is None
            else self.front_clearance
        )
        rear = (
            Params.swing_height if self.rear_clearance is None else self.rear_clearance
        )
        self.swing.swing_height = np.array([front, front, rear, rear])
        self.low_level.Kp = Params.Kp_swing
        # Walk, pace, bound and stand need damping on A1's stance joints.
        self.low_level.stance_damping = (
            np.maximum(1.0 - self.model.dof_damping[6:], 0.0)
            if self._active_gait
            in (GaitType.WALK, GaitType.PACE, GaitType.BOUNDING, GaitType.STAND)
            else np.zeros(12)
        )
        self.low_level.compensate_stance_bias = self._active_gait == GaitType.STAND
        self.mpc.last_contact = None
        self.mpc.roll_init = self.mpc.pitch_init = 0.0

    def __call__(self, model, data):
        """
        control callback.
        """
        if self.gait.gait_type != self._active_gait:
            self._configure_gait()
        self._filter_commands()
        self.robot_data.update()

        self.swing.update(self.v_des_current, self.yaw_rate_des_current)
        contact_forces = self.mpc.update(
            self.iter_counter,
            self.v_des_current,
            self.yaw_rate_des_current,
            self.gait.contact_schedule(),
        )
        torque_cmds = self.low_level.update(data, contact_forces)

        data.ctrl[:] = torque_cmds

        self.gait.update(Params.dt_control)
        self.iter_counter += 1
