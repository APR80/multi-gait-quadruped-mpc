"""Held-key controls; callbacks only change input state, never MuJoCo data."""
from mujoco.glfw import glfw
import numpy as np
from gait_generator import GaitType


GAIT_KEYS = {
    glfw.KEY_1: GaitType.STAND,
    glfw.KEY_2: GaitType.WALK,
    glfw.KEY_3: GaitType.TROT,
    glfw.KEY_4: GaitType.PACE,
    glfw.KEY_5: GaitType.BOUNDING,
    glfw.KEY_6: GaitType.AMBLE,
}


class KeyboardControl:
    def __init__(self, speed=.7):
        self.speed = speed
        self.held = set()
        self.gait_request = None
        self.reset_requested = False
        self.paused = False
        self.follow_camera = True
        self.show_overlay = True

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            self.held.discard(key)
            return
        if action != glfw.PRESS:
            return
        self.held.add(key)
        if key in GAIT_KEYS:
            self.gait_request = GAIT_KEYS[key]
        elif key == glfw.KEY_SPACE:
            self.held.clear()
        elif key == glfw.KEY_P:
            self.paused = not self.paused
            self.held.clear()
        elif key == glfw.KEY_R:
            self.reset_requested = True
            self.held.clear()
        elif key == glfw.KEY_F:
            self.follow_camera = not self.follow_camera
        elif key == glfw.KEY_H:
            self.show_overlay = not self.show_overlay
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.speed = min(1.5, round(self.speed + .1, 2))
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.speed = max(.1, round(self.speed - .1, 2))

    def on_focus(self, window, focused):
        if not focused:
            self.held.clear()

    def command(self):
        if self.paused:
            return np.zeros(3), 0.
        forward = int(glfw.KEY_W in self.held) - int(glfw.KEY_S in self.held)
        left = int(glfw.KEY_A in self.held) - int(glfw.KEY_D in self.held)
        turn = int(glfw.KEY_Q in self.held) - int(glfw.KEY_E in self.held)
        velocity = np.array([forward * self.speed, left * min(self.speed, .4), 0.])
        # Diagonal input keeps the same total translational speed.
        norm = np.linalg.norm(velocity)
        if norm > self.speed:
            velocity *= self.speed / norm
        return velocity, turn * .6

    def overlay(self, controller):
        gait = controller.gait.gait_type.value
        if controller.gait.pending_gait is not None:
            gait += " -> " + controller.gait.pending_gait.value + " (settling)"
        velocity, yaw = self.command()
        left = ("Gait\nDrive speed / gait limit\nCommand vx / vy / yaw\n"
                "Move\nTurn\nSpeed\nGaits\n\nStop / Pause / Reset\nCamera / Overlay / Exit")
        right = (f"{gait}{' [PAUSED]' if self.paused else ''}\n"
                 f"{self.speed:.1f} / {controller.gait.max_speed:.1f} m/s\n{velocity[0]:+.1f} / {velocity[1]:+.1f} / {yaw:+.1f}\n"
                 "Hold W/S forward/back, A/D left/right\nHold Q/E\n+ / -\n"
                 "1 Stand  2 Walk  3 Trot\n4 Pace  5 Bounding  6 Amble\n"
                 "Space / P / R\nF follow / H hide / Esc close")
        return left, right
