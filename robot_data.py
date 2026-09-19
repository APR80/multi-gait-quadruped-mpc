import mujoco as mj
import numpy as np
from parameters import Params


class RobotData:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.trunk_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "trunk")

        self.foot_geom_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.thigh_body_names = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]

        self.foot_geom_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
            for name in self.foot_geom_names
        ]
        self.thigh_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            for name in self.thigh_body_names
        ]
        # The foot sites mark sphere centers, so touchdown is one radius above
        # the flat floor. A1's site size differs from its collision radius.
        self.foot_radii = np.empty(4)
        for i, site_id in enumerate(self.foot_geom_ids):
            candidates = np.flatnonzero(
                (model.geom_bodyid == model.site_bodyid[site_id])
                & (model.geom_type == mj.mjtGeom.mjGEOM_SPHERE)
                & np.all(np.isclose(model.geom_pos, model.site_pos[site_id]), axis=1)
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one collision sphere for {self.foot_geom_names[i]}"
                )
            self.foot_radii[i] = model.geom_size[candidates[0], 0]

        # Pre-allocate Jacobian buffers to avoid GC pressure
        self._jacp_buf = np.zeros((3, self.model.nv))
        self._jacr_buf = np.zeros((3, self.model.nv))

        # Pre-allocate the full Jacobian storage (4 legs, 3 dims, Nv joints)
        self.Jv_feet = np.zeros((4, 3, self.model.nv))
        self.yaw = 0

    def update(self):
        self.pos_base = self.data.xpos[self.trunk_id].copy()
        self.lin_vel_base = self.data.qvel[0:3].copy()
        self.quat_base = self.data.sensordata[0:4].copy()
        self.ang_vel_base = self.data.sensordata[4:7].copy()
        self.q = self.data.sensordata[7:19].copy()
        self.qdot = self.data.sensordata[19:31].copy()

        self.R_base = self.data.xmat[self.trunk_id].reshape(3, 3).copy()
        mj.mj_subtreeVel(self.model, self.data)
        self.pos_com = self.data.subtree_com[self.trunk_id].copy()
        self.lin_vel_com = self.data.subtree_linvel[self.trunk_id].copy()

        self.pos_feet = self._compute_foot_positions_world()
        self.pos_feet_rel_world = self._compute_foot_positions_relative_world()
        self.pos_feet_rel_base = self._compute_foot_positions_relative_base_frame()

        self.pos_thighs = self._compute_thigh_positions_world()
        self.pos_thighs_rel_base = self._compute_thigh_positions_relative_base_frame()

        self.__update_jacobian()
        self.vel_foot_rel_base = self._compute_foot_velocities_relative_base_frame()

        self.yaw = np.atan2(
            2
            * (
                self.quat_base[0] * self.quat_base[3]
                + self.quat_base[1] * self.quat_base[2]
            ),
            1 - 2 * (self.quat_base[2] ** 2 + self.quat_base[3] ** 2),
        )

    def _compute_foot_positions_world(self):
        """(4, 3) array of foot positions in World Frame."""
        return self.data.site_xpos[self.foot_geom_ids]

    def _compute_foot_positions_relative_world(self):
        """(4, 3) array of vectors from Base to Feet in World Orientation."""
        # self.pos_feet is (4, 3), self.pos_base is (3,)
        # NumPy subtracts the (3,) from every row of the (4, 3) automatically.
        return self.pos_feet - self.pos_base

    def _compute_foot_positions_relative_base_frame(self):
        """Returns (4, 3) array of Foot positions relative to Base, in Base Frame."""
        # (4, 3) @ (3, 3) -> (4, 3)
        # This rotates all 4 vectors into the local frame simultaneously.
        return (self.pos_feet - self.pos_base) @ self.R_base

    def _compute_thigh_positions_world(self):
        # This returns a (4, 3) array containing all 4 thigh positions
        return self.data.xpos[self.thigh_ids]

    def _compute_thigh_positions_relative_base_frame(self):
        # self.pos_thighs is (4, 3), self.pos_base is (3,)
        return (self.pos_thighs - self.pos_base) @ self.R_base

    def __update_jacobian(self):

        for i, geom_id in enumerate(self.foot_geom_ids):
            mj.mj_jacSite(
                self.model, self.data, self._jacp_buf, self._jacr_buf, geom_id
            )
            self.Jv_feet[i] = self._jacp_buf  # Store world-frame linear Jacobian

    def _compute_foot_velocities_relative_base_frame(self):
        # generalized_qdot: (nv,)
        # self.Jv_feet: (4, 3, nv)
        # Resulting vel_feet_world: (4, 3)
        vel_feet_world = self.Jv_feet @ self.data.qvel

        # Calculate Base-Relative Velocities in Base-Frame (Vectorized)
        # (4, 3) @ (3, 3) -> (4, 3)
        return (
            vel_feet_world
            - self.lin_vel_base
            - np.cross(self.ang_vel_base, self.pos_feet_rel_base)
        ) @ self.R_base

    def get_state(self):

        q = self.quat_base
        roll = np.atan2(
            2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        )
        pitch = np.asin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0))

        return np.concatenate(
            [
                np.array([roll, pitch, self.yaw]),  # [0:3]   Roll, Pitch, Yaw
                self.pos_com,  # [3:6]   whole-robot CoM X, Y, Z
                self.ang_vel_base,  # [6:9]   wx, wy, wz
                self.lin_vel_com,  # [9:12]  CoM vx, vy, vz
                np.array([-Params.gravity]),
            ]
        )
