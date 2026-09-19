"""Run the quadruped viewer or a reproducible headless simulation."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path
import time

import mujoco as mj
import numpy as np

from locomotion_controller import LocomotionController
from gait_generator import GaitType


SCENES = {
    "go1": "go1/scene.xml",
    "a1": "unitree_robotics_a1/scene.xml",
}


def create_simulation(
    robot="a1", gait="trot", front_clearance=None, rear_clearance=None
):
    path = Path(__file__).resolve().parent / SCENES[robot]
    model = mj.MjModel.from_xml_path(str(path))
    data = mj.MjData(model)
    controller = LocomotionController(
        model, data, gait, front_clearance, rear_clearance
    )
    return model, data, controller


def run_headless(model, data, controller, duration):
    """Simulate for a fixed duration and report stability/tracking metrics."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    heights, tilts, velocities, yaw_rates = [], [], [], []
    max_torque = 0.0
    fell = False
    previous_swing = np.zeros(4, dtype=bool)
    swing_peaks = np.zeros(4)
    swing_peak_sum = np.zeros(4)
    swing_counts = np.zeros(4, dtype=int)
    slip_sum = np.zeros(4)
    stance_counts = np.zeros(4, dtype=int)
    start = data.time
    mj.set_mjcb_control(controller)
    try:
        for _ in range(int(np.ceil(duration / model.opt.timestep))):
            mj.mj_step(model, data)
            state = controller.robot_data.get_state()
            height = float(data.qpos[2])
            tilt = float(np.max(np.abs(state[:2])))
            heights.append(height)
            tilts.append(tilt)
            max_torque = max(max_torque, float(np.max(np.abs(data.ctrl))))
            if data.time - start >= min(2.0, duration / 2):
                # Planar tracking is measured in the heading frame.
                yaw = controller.robot_data.yaw
                c, s = np.cos(yaw), np.sin(yaw)
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                velocities.append(controller.robot_data.lin_vel_base @ rotation)
                yaw_rates.append(float(controller.robot_data.ang_vel_base[2]))
                robot = controller.robot_data
                swing = controller.swing.prev_swing
                clearance = robot.pos_feet[:, 2] - robot.foot_radii
                touchdown = previous_swing & ~swing
                swing_peak_sum[touchdown] += swing_peaks[touchdown]
                swing_counts[touchdown] += 1
                swing_peaks[swing & ~previous_swing] = 0.0
                swing_peaks[swing] = np.maximum(swing_peaks[swing], clearance[swing])
                world_velocity = (
                    robot.base_vel_base_feet @ robot.R_base.T
                    + robot.lin_vel_base
                    + np.cross(robot.ang_vel_base, robot.pos_feet_rel_world)
                )
                planted = ~swing & (clearance < 0.005)
                slip_sum[planted] += np.linalg.norm(world_velocity[planted, :2], axis=1)
                stance_counts[planted] += 1
                previous_swing = swing.copy()
            if not np.all(np.isfinite(data.qpos)) or height < 0.15 or tilt > 0.8:
                fell = True
                break
    finally:
        mj.set_mjcb_control(None)
    return {
        "simulated_seconds": float(data.time - start),
        "fell": fell,
        "min_base_height_m": min(heights),
        "max_base_height_m": max(heights),
        "max_abs_roll_pitch_rad": max(tilts),
        "mean_heading_velocity_m_s": np.mean(velocities, axis=0).tolist()
        if velocities
        else None,
        "mean_yaw_rate_rad_s": float(np.mean(yaw_rates)) if yaw_rates else None,
        "max_abs_joint_torque_nm": max_torque,
        "mpc_solves": controller.mpc.solve_count,
        "mpc_failures": controller.mpc.failure_count,
        "mujoco_warnings": int(sum(warning.number for warning in data.warning)),
        "final_position_m": data.qpos[:3].tolist(),
        "gait": controller.gait.gait_type.value,
        "requested_velocity_m_s": np.asarray(
            controller.v_des_target, dtype=float
        ).tolist(),
        "gait_speed_limit_m_s": controller.gait.max_speed,
        "mean_swing_clearance_m": (
            swing_peak_sum / np.maximum(swing_counts, 1)
        ).tolist(),
        "completed_swings_per_foot": swing_counts.tolist(),
        "mean_stance_slip_m_s": (slip_sum / np.maximum(stance_counts, 1)).tolist(),
    }


def run_keyboard(model, data, controller, args):
    from keyboard_control import KeyboardControl
    from renderer import MujocoRenderer
    from mujoco.glfw import glfw

    keyboard = KeyboardControl(args.speed)
    keyboard.show_overlay = not args.no_overlay
    viewer = MujocoRenderer(
        model,
        data,
        title=f"{args.robot.upper()} - Keyboard locomotion",
        key_callback=keyboard.on_key,
        focus_callback=keyboard.on_focus,
    )
    steps_per_frame = max(1, round(1 / (60 * model.opt.timestep)))
    frame_time = steps_per_frame * model.opt.timestep
    elapsed = 0.0
    print("Hold WASD to move, Q/E to turn; release to stop. +/- changes speed.")
    print(
        "1 stand, 2 walk, 3 trot, 4 pace, 5 bounding, 6 amble. Space stop, P pause, R reset, F camera, H overlay, Esc exit."
    )
    mj.set_mjcb_control(controller)
    try:
        while not viewer.is_window_closed() and (
            args.duration is None or elapsed < args.duration
        ):
            start = time.monotonic()
            glfw.poll_events()
            if keyboard.reset_requested:
                gait = (
                    keyboard.gait_request
                    or controller.gait.pending_gait
                    or controller.gait.gait_type
                )
                mj.set_mjcb_control(None)
                controller = LocomotionController(
                    model, data, gait, args.front_clearance, args.rear_clearance
                )
                mj.set_mjcb_control(controller)
                keyboard.reset_requested = False
            if keyboard.gait_request is not None:
                controller.request_gait(keyboard.gait_request)
                keyboard.gait_request = None
            controller.v_des_target, controller.yaw_rate_des_target = keyboard.command()
            if not keyboard.paused:
                for _ in range(steps_per_frame):
                    mj.mj_step(model, data)
                    elapsed += model.opt.timestep
                    if args.duration is not None and elapsed >= args.duration:
                        break
            if keyboard.follow_camera:
                viewer.cam.lookat[:] = data.body("trunk").xpos
            viewer.overlay_text = (
                keyboard.overlay(controller) if keyboard.show_overlay else None
            )
            viewer.render()
            remaining = frame_time - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        mj.set_mjcb_control(None)
        viewer.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=SCENES, default="go1")
    parser.add_argument("--gait", choices=[g.value for g in GaitType], default="trot")
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="hold WASD/QE to drive; number keys select gaits",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="hide all keyboard overlay text; H toggles it",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.7,
        help="keyboard translation speed in m/s (0.1 to 1.5)",
    )
    parser.add_argument(
        "--front-clearance", type=float, help="front swing height in metres"
    )
    parser.add_argument(
        "--rear-clearance", type=float, help="rear swing height in metres"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, help="seconds (headless default: 10)")
    parser.add_argument("--vx", type=float, default=0.7, help="forward velocity in m/s")
    parser.add_argument(
        "--vy", type=float, default=0.0, help="leftward velocity in m/s"
    )
    parser.add_argument(
        "--yaw-rate", type=float, default=0.0, help="turn rate in rad/s"
    )
    args = parser.parse_args()
    if args.keyboard and args.headless:
        parser.error("--keyboard requires a viewer; omit --headless")
    if not np.isfinite(args.speed) or not 0.1 <= args.speed <= 1.5:
        parser.error("--speed must be between 0.1 and 1.5 m/s")
    for name in ("front_clearance", "rear_clearance"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or not 0.02 <= value <= 0.14):
            parser.error(f"--{name.replace('_', '-')} must be between 0.02 and 0.14 m")
    if args.duration is not None and (
        not np.isfinite(args.duration) or args.duration <= 0
    ):
        parser.error("--duration must be positive and finite")
    if not np.all(np.isfinite([args.vx, args.vy, args.yaw_rate])):
        parser.error("velocity commands must be finite")
    model, data, controller = create_simulation(
        args.robot, args.gait, args.front_clearance, args.rear_clearance
    )
    if args.keyboard:
        return run_keyboard(model, data, controller, args)
    controller.v_des_target = np.array([args.vx, args.vy, 0.0])
    controller.yaw_rate_des_target = args.yaw_rate
    if args.headless:
        summary = run_headless(model, data, controller, args.duration or 10.0)
        print(json.dumps(summary, indent=2))
        return int(
            summary["fell"] or summary["mpc_failures"] or summary["mujoco_warnings"]
        )

    import mujoco.viewer

    mj.set_mjcb_control(controller)
    try:
        with mj.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and (
                args.duration is None or data.time < args.duration
            ):
                start = time.monotonic()
                mj.mj_step(model, data)
                viewer.sync()
                remaining = model.opt.timestep - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        mj.set_mjcb_control(None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
