from dataclasses import dataclass
from enum import Enum
import numpy as np
from parameters import Params


class GaitType(Enum):
    STAND = "stand"
    WALK = "walk"
    TROT = "trot"
    PACE = "pace"
    BOUNDING = "bounding"
    AMBLE = "amble"


@dataclass(frozen=True)
class GaitProfile:
    period: float
    duty: float
    offsets: tuple
    max_speed: float
    stance_width_scale: float = 1.0


GAIT_PROFILES = {
    GaitType.STAND: GaitProfile(1.0, 1.0, (0.0, 0.0, 0.0, 0.0), 0.0),
    GaitType.WALK: GaitProfile(0.48, 0.75, (0.0, 0.5, 0.75, 0.25), 0.3),
    GaitType.TROT: GaitProfile(0.3, 0.6, (0.5, 0.0, 0.0, 0.5), 1.2),
    GaitType.PACE: GaitProfile(
        0.36, 0.6, (0.5, 0.0, 0.5, 0.0), 0.7, stance_width_scale=1.0
    ),
    GaitType.BOUNDING: GaitProfile(0.28, 0.55, (0.5, 0.5, 0.0, 0.0), 1.2),
    GaitType.AMBLE: GaitProfile(0.4, 0.65, (0.0, 0.5, 0.75, 0.25), 0.7),
}


class GaitGenerator:
    def __init__(self, gait_period=None, duty_factor=None, gait_type=GaitType.TROT):
        self._dt_control = Params.dt_control
        self._dt_mpc = Params.dt_mpc
        self._mpc_horizon = Params.horizon
        self._apply(GaitType(gait_type))
        if gait_period is not None:
            self.gait_period = gait_period
        if duty_factor is not None and self.gait_type != GaitType.STAND:
            self.duty_factor = duty_factor
        if self.gait_period <= 0 or not 0 < self.duty_factor <= 1:
            raise ValueError("Gait period must be positive and duty factor in (0, 1]")
        self.pending_gait = None
        self._transition_elapsed = 0.0
        self._transition_duration = 0.0

    def _apply(self, gait_type):
        profile = GAIT_PROFILES[gait_type]
        self.gait_type = gait_type
        self.gait_period = profile.period
        self.duty_factor = profile.duty
        self.phase_offset = np.array(profile.offsets, dtype=float)
        self.global_phase = 0.0

    @property
    def swing_time(self):
        return self.gait_period * (1 - self.duty_factor)

    @property
    def stance_time(self):
        return self.gait_period * self.duty_factor

    @property
    def max_speed(self):
        return GAIT_PROFILES[self.gait_type].max_speed

    @property
    def stance_width_scale(self):
        return GAIT_PROFILES[self.gait_type].stance_width_scale

    @property
    def transitioning(self):
        return self.pending_gait is not None

    def request_gait(self, gait_type):
        """Finish airborne steps, then start the new gait."""
        gait_type = GaitType(gait_type)
        if self.transitioning:
            self.pending_gait = gait_type
            return
        if gait_type == self.gait_type:
            return
        phase = (self.global_phase - self.phase_offset) % 1.0
        self._transition_phase = phase.copy()
        self._transition_swing = phase >= self.duty_factor
        self._landing_times = np.where(
            self._transition_swing, (1 - phase) * self.gait_period, 0.0
        )
        self._transition_elapsed = 0.0
        self._transition_duration = float(self._landing_times.max() + 0.18)
        self.pending_gait = gait_type

    def update(self, dt=None):
        dt = self._dt_control if dt is None else dt
        if self.transitioning:
            self._transition_elapsed += dt
            if self._transition_elapsed >= self._transition_duration:
                remainder = self._transition_elapsed - self._transition_duration
                self._apply(self.pending_gait)
                self.pending_gait = None
                self.global_phase = remainder / self.gait_period
        else:
            self.global_phase = (self.global_phase + dt / self.gait_period) % 1.0

    @staticmethod
    def _sample_profile(times, phase, period, duty, offsets):
        local = (phase + np.asarray(times)[:, None] / period - offsets) % 1.0
        swing = local >= duty
        progress = np.where(swing, (local - duty) / max(1 - duty, 1e-12), 0.0)
        return swing, progress

    def _sample(self, times):
        times = np.asarray(times, dtype=float)
        if not self.transitioning:
            return self._sample_profile(
                times,
                self.global_phase,
                self.gait_period,
                self.duty_factor,
                self.phase_offset,
            )
        elapsed = self._transition_elapsed + times
        swing = self._transition_swing & (elapsed[:, None] < self._landing_times)
        local = self._transition_phase + elapsed[:, None] / self.gait_period
        progress = np.where(
            swing, (local - self.duty_factor) / max(1 - self.duty_factor, 1e-12), 0.0
        )
        after = elapsed >= self._transition_duration
        if np.any(after):
            profile = GAIT_PROFILES[self.pending_gait]
            swing[after], progress[after] = self._sample_profile(
                elapsed[after] - self._transition_duration,
                0.0,
                profile.period,
                profile.duty,
                np.array(profile.offsets),
            )
        return swing, progress

    def contact_schedule(self):
        # state references start one sample later.
        swing, _ = self._sample(np.arange(self._mpc_horizon) * self._dt_mpc)
        return (~swing).astype(float).flatten()

    def get_swing_state(self):
        return self._sample([0.0])[1][0]

    def get_swing_contact(self):
        return self._sample([0.0])[0][0]

    def get_stance_state(self):
        phase = (self.global_phase - self.phase_offset) % 1.0
        return np.where(self.get_swing_contact(), 0.0, phase / self.duty_factor)
