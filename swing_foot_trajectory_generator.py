import numpy as np
from parameters import Params


class SwingFootTrajectoryGenerator:
    """World-frame swings with stationary takeoff and touchdown endpoints."""

    def __init__(self, robot_data, gait_generator):
        self.robot_data = robot_data
        self.gait_generator = gait_generator
        self.swing_height = Params.swing_height
        self.apex_phase = Params.swing_apex_phase
        self.p0 = np.zeros((4, 3))
        self.p5 = np.zeros((4, 3))
        self.ground_height = np.asarray(robot_data.foot_radii, dtype=float).copy()
        self.prev_swing = np.zeros(4, dtype=bool)
        self._initialized = False
        self.pos_des_base = np.zeros((4, 3))
        self.vel_des_base = np.zeros((4, 3))

    def nominal_footholds_base(self):
        """Place feet under the thighs, with the gait's lateral spacing."""
        footholds = self.robot_data.pos_thighs_rel_base.copy()
        footholds[:, 1] *= self.gait_generator.stance_width_scale
        return footholds

    @staticmethod
    def _quintic(s):
        """Quintic blend with its phase derivative."""
        return s**3 * (10 + s * (-15 + 6 * s)), 30 * s**2 * (1 - s) ** 2

    @staticmethod
    def _raised_cosine(s):
        """Blend with its phase derivative."""
        return 0.5 * (1 - np.cos(np.pi * s)), 0.5 * np.pi * np.sin(np.pi * s)

    def _trajectory(self, phases, swing_time):
        s = np.clip(np.asarray(phases), 0, 1)[:, None]
        blend, derivative = self._quintic(s)
        span = self.p5 - self.p0
        position = self.p0 + blend * span
        velocity = derivative * span / swing_time

        apex_phase = float(np.clip(self.apex_phase, 1e-3, 1 - 1e-3))
        ascending = s[:, 0] < apex_phase
        segment = np.where(
            ascending, s[:, 0] / apex_phase, (s[:, 0] - apex_phase) / (1 - apex_phase)
        )
        rate = np.where(ascending, 1 / apex_phase, 1 / (1 - apex_phase))
        lift, drop = self._raised_cosine(segment), self._quintic(segment)
        apex = np.maximum(self.p0[:, 2], self.p5[:, 2]) + self.swing_height
        z_start = np.where(ascending, self.p0[:, 2], apex)
        z_span = np.where(ascending, apex, self.p5[:, 2]) - z_start
        position[:, 2] = z_start + np.where(ascending, lift[0], drop[0]) * z_span
        velocity[:, 2] = (
            np.where(ascending, lift[1], drop[1]) * z_span * rate / swing_time
        )
        return position, velocity

    def _track_ground_height(self, is_swing):
        """Follow the height a loaded foot actually rests at."""
        planted = ~np.asarray(is_swing, dtype=bool)
        if np.any(planted):
            measured = self.robot_data.pos_feet[planted, 2]
            self.ground_height[planted] += Params.ground_height_filter * (
                measured - self.ground_height[planted]
            )
        return self.ground_height

    def update(self, v_cmd_body, yaw_rate_cmd):
        phases = self.gait_generator.get_swing_state()
        is_swing = self.gait_generator.get_swing_contact()
        swing_time = max(self.gait_generator.swing_time, 1e-6)
        stance_time = self.gait_generator.stance_time
        robot = self.robot_data
        if not self._initialized:
            self.p0[:] = robot.pos_feet
            self.p5[:] = robot.pos_feet
            self._initialized = True
        ground = self._track_ground_height(is_swing)

        # Capture the actual liftoff position on entry to swing, including the
        # valid phase-zero sample. Capturing at touchdown misses stance travel.
        new_swing = is_swing & ~self.prev_swing
        if np.any(new_swing):
            self.p0[new_swing] = robot.pos_feet[new_swing]
            v_cmd_world = self._rotation_z(robot.yaw) @ v_cmd_body
            yaw_comp = self._rotation_z(yaw_rate_cmd * 0.5 * stance_time)
            hip_local = self.nominal_footholds_base() @ yaw_comp.T
            hip_world = robot.pos_base + hip_local @ robot.R_base.T
            remaining = ((1.0 - phases) * swing_time)[:, None]
            raibert = robot.lin_vel_base * stance_time * 0.5 + 0.03 * (
                robot.lin_vel_base - v_cmd_world
            )
            turn = 0.5 * robot.pos_base[2] / Params.gravity * yaw_rate_cmd
            centrifugal = turn * np.array(
                [robot.lin_vel_base[1], -robot.lin_vel_base[0], 0]
            )
            targets = hip_world + v_cmd_world * remaining + raibert + centrifugal
            targets[:, 2] = ground
            self.p5[new_swing] = targets[new_swing]

        position, velocity = self._trajectory(phases, swing_time)
        relative_position = position - robot.pos_base
        self.pos_des_base = relative_position @ robot.R_base
        self.vel_des_base = (
            velocity
            - robot.lin_vel_base
            - np.cross(robot.ang_vel_base, relative_position)
        ) @ robot.R_base
        self.prev_swing = is_swing.copy()

    def predict_moment_arms(self, contacts, dt, velocity):
        """Predict stance anchors across touchdown and subtract future CoM."""
        contacts = np.asarray(contacts).reshape(-1, 4).astype(bool)
        robot = self.robot_data
        nominal_world = robot.pos_base + self.nominal_footholds_base() @ robot.R_base.T
        anchor = robot.pos_feet.copy()
        first_landing = self.gait_generator.get_swing_contact().copy()
        arms = np.empty((len(contacts), 4, 3))
        for k, contact in enumerate(contacts):
            if k:
                touchdown = contact & ~contacts[k - 1]
                planned = (
                    nominal_world
                    + velocity * (k * dt + 0.5 * self.gait_generator.stance_time)
                    + 0.03 * (robot.lin_vel_base - velocity)
                )
                planned[:, 2] = self.ground_height
                planned[first_landing] = self.p5[first_landing]
                anchor[touchdown] = planned[touchdown]
                first_landing[touchdown] = False
            arms[k] = anchor - (robot.pos_com + velocity * k * dt)
        return arms

    @staticmethod
    def _rotation_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
