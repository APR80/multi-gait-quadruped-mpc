import numpy as np
import mujoco as mj


class Params:
    dt_control = 0.004
    iteration_between_mpc = 10
    dt_mpc: float = dt_control * iteration_between_mpc

    horizon: int = 16

    gravity = 9.81

    friction_coef = 0.7

    # quadratic programming weights
    # r, p, y, x, y, z, wx, wy, wz, vx, vy, vz, g
    Q = np.diag([5.0, 5.0, 10.0, 10.0, 10.0, 50.0, 0.01, 0.01, 0.2, 0.2, 0.2, 0.2, 0.0])
    R = np.diag(
        [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
    )

    fz_max = 500.0
    base_height_des: float = 0.30

    swing_height = 0.035

    # A bound pitches the body so the front feet swing under a rising trunk and need the extra room.
    bound_front_clearance = 0.05

    # Fraction of the swing spent lifting.
    swing_apex_phase = 0.45

    # Blend rate for the tracked ground height
    ground_height_filter = 0.2
    Kp_swing = np.diag([400.0, 400.0, 400.0])
    Kd_swing = np.diag([60.0, 60.0, 60.0])

    @classmethod
    def init_from_model(cls, model, body_name="trunk", data=None):

        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 1:
            raise ValueError(f"Robot body {body_name!r} was not found")
        if data is None:
            data = mj.MjData(model)
            home = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
            if home >= 0:
                mj.mj_resetDataKeyframe(model, data, home)
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)

        subtree = np.zeros(model.nbody, dtype=bool)
        subtree[body_id] = True
        for i in range(body_id + 1, model.nbody):
            subtree[i] = subtree[model.body_parentid[i]]
        masses = model.body_mass[subtree]
        cls.mass = float(masses.sum())
        com = np.average(data.xipos[subtree], axis=0, weights=masses)
        offsets = data.xipos[subtree] - com
        rotations = data.ximat[subtree].reshape(-1, 3, 3)
        inertia_world = np.zeros((3, 3))
        for mass, offset, rotation, diagonal in zip(
            masses, offsets, rotations, model.body_inertia[subtree]
        ):
            inertia_world += rotation @ np.diag(diagonal) @ rotation.T
            inertia_world += mass * (
                np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
            )
        rotation_base = data.xmat[body_id].reshape(3, 3)
        cls.inertia = rotation_base.T @ inertia_world @ rotation_base
        cls.com_offset_base = rotation_base.T @ (com - data.xpos[body_id])
