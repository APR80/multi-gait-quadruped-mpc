from functools import partial

# must be at the top
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import osqp
from scipy import sparse
from parameters import Params


# --- JAX JIT Compiled Kernels ---
@jax.jit
def vec2so3_jax(v):
    """skew-symmetric matrix generation."""
    return jnp.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=jnp.float64
    )


@jax.jit
def get_discretized_dynamics(yaw, inv_base_inertia, pos_base_feet, mass, dt):
    """
    Computes continuous state-space matrices and discretizes them.
    Fully fused kernel.
    """
    num_state = 13
    num_input = 12

    # Continuous Dynamics
    Ac = jnp.zeros((num_state, num_state), dtype=jnp.float64)
    Bc = jnp.zeros((num_state, num_input), dtype=jnp.float64)

    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    Rz = jnp.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)

    inv_world_I = Rz @ inv_base_inertia @ Rz.T

    Ac = Ac.at[0:3, 6:9].set(Rz.T)
    Ac = Ac.at[3:6, 9:12].set(jnp.eye(3))
    Ac = Ac.at[11, 12].set(1.0)

    for i in range(4):
        Bc = Bc.at[6:9, 3 * i : 3 * i + 3].set(
            inv_world_I @ vec2so3_jax(pos_base_feet[i])
        )
        Bc = Bc.at[9:12, 3 * i : 3 * i + 3].set(jnp.eye(3) / mass)

    # Exact ZOH
    Ad = jnp.eye(num_state) + dt * Ac + 0.5 * dt**2 * (Ac @ Ac)
    Bd = dt * Bc + 0.5 * dt**2 * (Ac @ Bc)

    return Ad, Bd


@jax.jit
def get_horizon_dynamics(yaw, inv_inertia, moment_arms, mass, dt):
    Ad, Bd = jax.vmap(get_discretized_dynamics, in_axes=(None, None, 0, None, None))(
        yaw, inv_inertia, moment_arms, mass, dt
    )
    return Ad[0], Bd


@partial(jax.jit, static_argnums=(6,))
def construct_qp_cost(Ad, Bd, Qbar, Rbar, xt, Xref, horizon):
    num_state = Ad.shape[0]
    num_input = Bd.shape[-1]

    def power_scan(carry, _):
        next_A = carry @ Ad
        return next_A, next_A

    _, powers = jax.lax.scan(power_scan, jnp.eye(num_state), None, length=horizon)
    Sx = jnp.concatenate(powers, axis=0)

    Su = jnp.zeros((num_state * horizon, num_input * horizon))
    for i in range(horizon):
        for j in range(i + 1):
            B_step = Bd[j] if Bd.ndim == 3 else Bd
            idx_row = slice(i * num_state, (i + 1) * num_state)
            idx_col = slice(j * num_input, (j + 1) * num_input)
            if i == j:
                Su = Su.at[idx_row, idx_col].set(B_step)
            else:
                Su = Su.at[idx_row, idx_col].set(powers[i - j - 1] @ B_step)

    H = 2.0 * (Su.T @ Qbar @ Su + Rbar)

    # regularization to guarantee positive def
    H = H + 1e-6 * jnp.eye(H.shape[0])
    g = 2.0 * Su.T @ Qbar @ (Sx @ xt - Xref)
    return H, g


# --- Main Controller Class ---
class ConvexMPC:
    def __init__(self, robot_data):
        self.robot_data = robot_data
        self.num_state = 13
        self.num_input = 12
        self.num_constraints_per_step = 20

        self.is_initialized = False
        self.last_contact = None
        self.last_status = None
        self.solve_count = 0
        self.failure_count = 0

        self.last_solution_x = None
        self.last_solution_y = None
        self.__contact_forces = np.zeros(12)

        self._load_parameters()
        self._init_osqp()

        self.xpos_base_desired = 0
        self.ypos_base_desired = 0
        self.yaw_desired = 0.0
        self.roll_init = 0.0
        self.pitch_init = 0.0

    def _load_parameters(self):
        self.dt_control = Params.dt_control
        self.iterations_between_mpc = Params.iteration_between_mpc
        self.dt = Params.dt_mpc
        self.horizon = Params.horizon
        self.mu = Params.friction_coef
        self.fz_max = Params.fz_max
        self.gravity = Params.gravity

        self.mass = float(Params.mass)
        self.com_height_des = float(Params.base_height_des + Params.com_offset_base[2])

        self.base_inertia_base = jnp.array(Params.inertia, dtype=jnp.float64)
        self.inv_base_inertia = jnp.linalg.inv(self.base_inertia_base)

        self.Qbar = jnp.kron(
            jnp.identity(self.horizon), jnp.array(Params.Q, dtype=jnp.float64)
        )
        self.Rbar = jnp.kron(
            jnp.identity(self.horizon), jnp.array(Params.R, dtype=jnp.float64)
        )

    def _init_osqp(self):
        self.solver = osqp.OSQP()
        self.nv = self.num_input * self.horizon
        self.nc = self.num_constraints_per_step * self.horizon

        # OSQP setup
        P_init = sparse.triu(np.ones((self.nv, self.nv)), format="csc")
        self._P_rows = P_init.indices.copy()
        self._P_cols = np.repeat(np.arange(self.nv), np.diff(P_init.indptr))
        P_init.data[:] = (self._P_rows == self._P_cols).astype(float)
        q_init = np.zeros(self.nv)

        # Pre-allocate A matrix
        self.A_sparse, l_init, u_init = self._init_QP_constraints(
            np.ones(4 * self.horizon)
        )

        self.solver.setup(
            P=P_init,
            q=q_init,
            A=self.A_sparse,
            l=l_init,
            u=u_init,
            warm_starting=True,
            verbose=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=4000,
            sigma=1e-6,
            adaptive_rho=True,
        )

    def _init_QP_constraints(self, gait_table):
        """Builds the Constant Sparse A Matrix and Initial Bounds."""
        mu = self.mu
        single_foot_constraint = np.array(
            [[1, 0, mu], [-1, 0, mu], [0, 1, mu], [0, -1, mu], [0, 0, 1]],
            dtype=np.float64,
        )

        A_block = sparse.block_diag(
            [single_foot_constraint] * (4 * self.horizon), format="csc"
        )
        l_bounds, u_bounds = self._generate_QP_bounds(gait_table)

        return A_block, l_bounds, u_bounds

    def _generate_QP_bounds(self, gait_table):
        """Vectorized Bounds Update (runs every MPC loop)."""
        l_bounds = np.zeros(self.nc)
        u_bounds = np.full(self.nc, np.inf)

        # z-force constraints are located at index: (i*4 + j)*5 + 4
        z_force_indices = np.arange(4, self.nc, 5)
        u_bounds[z_force_indices] = gait_table * self.fz_max

        return l_bounds, u_bounds

    def _solve_mpc(self, ref_traj, gait_table):
        if hasattr(self, "swing") and self.swing._initialized:
            arms = self.swing.predict_moment_arms(
                gait_table, self.dt, self.velocity_world
            )
            Ad, Bd = get_horizon_dynamics(
                self.yaw, self.inv_base_inertia, jnp.array(arms), self.mass, self.dt
            )
        else:
            Ad, Bd = get_discretized_dynamics(
                self.yaw,
                self.inv_base_inertia,
                jnp.array(self.pos_base_feet),
                self.mass,
                self.dt,
            )

        # QP Cost
        H, g = construct_qp_cost(
            Ad,
            Bd,
            self.Qbar,
            self.Rbar,
            jnp.array(self.current_state),
            jnp.array(ref_traj),
            self.horizon,
        )

        # QP Bounds (Vectorized)
        l_qp, u_qp = self._generate_QP_bounds(gait_table)

        # Cast to Float64 / NumPy Arrays for OSQP
        H_np = np.array(H, dtype=np.float64)
        g_np = np.array(g, dtype=np.float64)

        self.solver.update(Px=H_np[self._P_rows, self._P_cols], q=g_np, l=l_qp, u=u_qp)

        # Warm Start
        if self.last_solution_x is not None:
            x_guess = np.roll(self.last_solution_x, -self.num_input)
            x_guess[-self.num_input :] = 0
            y_guess = np.roll(self.last_solution_y, -self.num_constraints_per_step)
            y_guess[-self.num_constraints_per_step :] = 0
            self.solver.warm_start(x=x_guess, y=y_guess)

        results = self.solver.solve(raise_error=False)
        self.last_status = results.info.status
        self.solve_count += 1

        if results.info.status in ("solved", "solved inaccurate") and np.all(
            np.isfinite(results.x)
        ):
            self.last_solution_x = results.x
            self.last_solution_y = results.y
            return results.x
        else:
            self.failure_count += 1
            if self.last_solution_x is not None:
                fallback = np.concatenate(
                    (self.last_solution_x[self.num_input :], np.zeros(self.num_input))
                )
            else:
                contacts = np.asarray(gait_table).reshape(self.horizon, 4)
                fallback = np.zeros((self.horizon, 4, 3))
                fallback[:, :, 2] = (
                    contacts
                    * self.mass
                    * self.gravity
                    / np.maximum(contacts.sum(axis=1, keepdims=True), 1)
                )
            return self._project_forces(fallback, gait_table).flatten()

    def _project_forces(self, forces, contacts):
        """Enforce contact and friction limits."""
        forces = np.asarray(forces).reshape(-1, 4, 3).copy()
        contacts = np.asarray(contacts).reshape(-1, 4)
        forces[:, :, 2] = np.clip(forces[:, :, 2], 0, contacts * self.fz_max)
        bound = self.mu * forces[:, :, 2, None]
        forces[:, :, :2] = np.clip(forces[:, :, :2], -bound, bound)
        return forces

    def _update_state_integration(self, vel_base_des, yaw_rate_des):
        # this runs at control rate, so integrate with dt_control.
        self.xpos_base_desired += vel_base_des[0] * self.dt_control
        self.ypos_base_desired += vel_base_des[1] * self.dt_control
        self.yaw_desired += yaw_rate_des * self.dt_control

    def update(self, iter_counter, base_vel_base_des, yaw_turn_rate_des, gait_table):
        self.current_state = self.robot_data.get_state()
        self.yaw = self.robot_data.yaw
        self.pos_base_feet = self.robot_data.pos_feet - self.robot_data.pos_com
        if not self.is_initialized:
            self.xpos_base_desired, self.ypos_base_desired = self.current_state[3:5]
            self.yaw_desired = self.yaw
            self.is_initialized = True
        # Keep the wrapped measured yaw on the same branch as the integrated target so crossing +/-pi does not request a full turn.
        self.current_state[2] = self.yaw_desired + np.arctan2(
            np.sin(self.yaw - self.yaw_desired), np.cos(self.yaw - self.yaw_desired)
        )

        R_z_ideal = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw), 0],
                [np.sin(self.yaw), np.cos(self.yaw), 0],
                [0, 0, 1],
            ]
        )
        vel_base_des_ideal = R_z_ideal @ base_vel_base_des
        self.velocity_world = vel_base_des_ideal

        contact_now = np.asarray(gait_table)[:4]
        contact_changed = not np.array_equal(contact_now, self.last_contact)
        if iter_counter % self.iterations_between_mpc == 0 or contact_changed:
            ref_traj = self.generate_reference_trajectory(
                vel_base_des_ideal, yaw_turn_rate_des
            )
            full_forces = self._solve_mpc(ref_traj, gait_table)
            self.__contact_forces = self._project_forces(
                full_forces[0:12], contact_now
            ).flatten()
            self.last_contact = contact_now.copy()

        self._update_state_integration(vel_base_des_ideal, yaw_turn_rate_des)
        return self.__contact_forces

    def generate_reference_trajectory(self, vel_base_des, yaw_turn_rate):
        """Horizon Reference Generator(Vectorized)."""
        max_pos_error = 0.1

        # bound position targets(preventing aggressive runaway integrations)
        cur_x = self.xpos_base_desired = np.clip(
            self.xpos_base_desired,
            self.current_state[3] - max_pos_error,
            self.current_state[3] + max_pos_error,
        )
        cur_y = self.ypos_base_desired = np.clip(
            self.ypos_base_desired,
            self.current_state[4] - max_pos_error,
            self.current_state[4] + max_pos_error,
        )

        if np.fabs(self.current_state[9]) > 0.2:
            self.pitch_init += (
                self.dt * (0.0 - self.current_state[1]) / self.current_state[9]
            )
        if np.fabs(self.current_state[10]) > 0.1:
            self.roll_init += (
                self.dt * (0.0 - self.current_state[0]) / self.current_state[10]
            )

        # Saturation for pitch and roll compensation
        self.roll_init = np.clip(self.roll_init, -0.25, 0.25)
        self.pitch_init = np.clip(self.pitch_init, -0.25, 0.25)
        roll_comp = self.current_state[10] * self.roll_init
        pitch_comp = self.current_state[9] * self.pitch_init

        X_ref = np.zeros((self.horizon, self.num_state), dtype=np.float64)
        i_steps = np.arange(1, self.horizon + 1)

        X_ref[:, 0] = roll_comp
        X_ref[:, 1] = pitch_comp
        X_ref[:, 2] = self.yaw_desired + i_steps * self.dt * yaw_turn_rate
        X_ref[:, 3] = cur_x + i_steps * self.dt * vel_base_des[0]
        X_ref[:, 4] = cur_y + i_steps * self.dt * vel_base_des[1]
        X_ref[:, 5] = self.com_height_des
        X_ref[:, 8] = yaw_turn_rate
        X_ref[:, 9] = vel_base_des[0]
        X_ref[:, 10] = vel_base_des[1]
        X_ref[:, 12] = -self.gravity

        return X_ref.flatten()
