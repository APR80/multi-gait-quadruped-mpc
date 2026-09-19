# multi-gait-quadruped-mpc

A multi-gait locomotion controller for quadruped robots, built on convex model predictive
control over the single rigid body model. Runs in MuJoCo on the Unitree Go1 and A1, with six
gaits selectable at runtime and smooth transitions between them.

[![Watch the video](https://img.youtube.com/vi/8v1ONeGRCd0/maxresdefault.jpg)](https://www.youtube.com/watch?v=8v1ONeGRCd0)

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

## Gaits
A `GaitProfile` carries a period, a duty factor, and four per-leg phase offsets; a single global phase
advances with the control clock, and each leg's contact state and swing progress are sampled
from it. Contact schedules for the MPC horizon come from sampling that same phase forward in
time, so the gait definition and the MPC's contact prediction can never drift apart.

Because the phase is continuous, gait changes don't have to snap to a segment boundary — the
generator settles into the requested gait over a transition window instead.

| Gait | Period (s) | Duty | Max speed (m/s) |
|---|---|---|---|
| Stand | 1.00 | 1.00 | — |
| Walk | 0.48 | 0.75 | 0.3 |
| Trot | 0.30 | 0.60 | 1.2 |
| Pace | 0.36 | 0.60 | 0.7 |
| Bounding | 0.28 | 0.55 | 1.2 |
| Amble | 0.40 | 0.65 | 0.7 |

## Usage

```bash
pip install mujoco numpy scipy osqp jax
```

```bash
python run_sim.py                                  # Go1, trotting
python run_sim.py --robot a1 --gait bounding
python run_sim.py --keyboard                       # drive it yourself
python run_sim.py --headless --duration 10 --vx 0.8 --yaw-rate 0.3
```

Useful flags: `--robot {go1,a1}`, `--gait {stand,walk,trot,pace,bounding,amble}`,
`--vx/--vy/--yaw-rate`, `--speed`, `--front-clearance`, `--rear-clearance`, `--no-overlay`.

### Keyboard

| Key | Action |
|---|---|
| `W`/`S`, `A`/`D` | drive forward/back, strafe left/right (hold) |
| `Q`/`E` | turn (hold) |
| `1`–`6` | stand, walk, trot, pace, bounding, amble |
| `+`/`-` | adjust drive speed |
| `Space` / `P` / `R` | stop / pause / reset |
| `F` / `H` / `Esc` | follow camera / hide overlay / quit |

## References
- **Di Carlo, J., Wensing, P. M., Katz, B., Bledt, G., & Kim, S. (2018). "Dynamic Locomotion
  in the MIT Cheetah 3 Through Convex Model-Predictive Control."** IROS 2018.
[MIT Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software)
  (BSD-3-Clause).
- **[pympc-quadruped](https://github.com/yinghansun/pympc-quadruped)** by yinghansun

## License
Robot models in `go1/` and `unitree_robotics_a1/` are Unitree Robotics assets, redistributed
under the BSD 3-Clause license in their respective directories.
