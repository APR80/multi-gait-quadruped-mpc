# multi-gait-quadruped-mpc

A multi-gait locomotion controller for quadruped robots, built on convex model predictive
control over the single rigid body model. Runs in MuJoCo on the Unitree Go1 and A1, with six
gaits selectable at runtime and smooth transitions between them.

## Stack

| Component | Implementation |
|---|---|
| **QP solver** | OSQP, with JAX for the condensed dynamics build |
| **Kinematics** | read directly from MuJoCo |
| **Simulator** | MuJoCo |
| **Gait generation** | phase-based — continuous global phase, per-leg offsets, period and duty factor |
| **Swing trajectory** | hand-rolled quintic + raised-cosine blends, vectorized `(4, 3)` across all legs |
| **Foothold prediction** | `predict_moment_arms` projects stance anchors across the full MPC horizon |
| **Ground tracking** | per-foot height estimate filtered from loaded stance feet |
