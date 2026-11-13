"""Minimal PV: replaceable output fractions, used power and explicit curtailment.

No panel/weather physics, RNG, battery decisions or facility power solver.
Profile segments are held constant; the owner maps scenario time to profile
time and splits again if its actual absorption limit changes inside a segment.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from peacsim.energy.config import PVConfig
from peacsim.energy.contracts import energy_kwh
from peacsim.metrics.tracking import _number, _seconds


PV_MODEL_VERSION = 'flexdc-pv-v1'


def _fraction(value):
    value = _number(value, 'output_fraction', True)
    if value > 1:
        raise ValueError('output_fraction must be in [0, 1]')
    return value


@dataclass(frozen=True)
class PVProfile:
    """Finite, explicitly sourced piecewise-constant normalized output curve.

    fractions[i] applies on [start + i*interval, start + (i+1)*interval).
    There is no implicit repeat, extrapolation or normalization of input data.
    """

    start_seconds: int
    interval_seconds: int
    fractions: tuple
    source: str

    def __post_init__(self):
        for name in ('start_seconds', 'interval_seconds'):
            object.__setattr__(self, name, _seconds(getattr(self, name), name))
        object.__setattr__(self, 'fractions', tuple(_fraction(v) for v in self.fractions))
        if self.interval_seconds == 0 or not self.fractions:
            raise ValueError('profile needs positive intervals and at least one fraction')
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('profile source must be a nonempty string')
        _seconds(self.end_seconds, 'profile end_seconds')

    @property
    def end_seconds(self):
        return self.start_seconds + self.interval_seconds * len(self.fractions)

    def segments(self, start_seconds, end_seconds):
        """Return (start, end, fraction) pieces covering exactly this window.

        Evaluate utilization separately on these pieces; averaging fractions
        before min(available, absorption) can overestimate used PV energy.
        """
        start = _seconds(start_seconds, 'window start_seconds')
        end = _seconds(end_seconds, 'window end_seconds')
        if end <= start or start < self.start_seconds or end > self.end_seconds:
            raise ValueError('positive window must be covered by the PV profile')
        pieces = []
        while start < end:
            index = (start - self.start_seconds) // self.interval_seconds
            stop = min(end, self.start_seconds + (index + 1) * self.interval_seconds)
            pieces.append((start, stop, self.fractions[index]))
            start = stop
        return tuple(pieces)


def load_pv_profile(path):
    """Read a replaceable normalized profile without changing its time scale."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return PVProfile(**data['profile'])


@dataclass(frozen=True)
class PVStep:
    duration_seconds: int
    output_fraction: float
    absorption_limit_kw: float
    pv_available_power_kw: float
    pv_used_power_kw: float
    pv_curtailed_power_kw: float
    pv_available_energy_kwh: float
    pv_used_energy_kwh: float
    pv_curtailed_energy_kwh: float


class PV:
    """Stateless facility-side PV calculation with an immutable configuration."""

    def __init__(self, config):
        if not isinstance(config, PVConfig):
            raise TypeError('config must be a PVConfig')
        self._config = config

    @property
    def config(self):
        return self._config

    def available_power_kw(self, output_fraction):
        return self._config.peak_power_kw * _fraction(output_fraction)

    def step(self, duration_seconds, *, output_fraction, absorption_limit_kw):
        """Settle one constant-fraction, constant-absorption segment.

        The facility supplies the actual PV absorption ceiling, including any
        executed battery charging. This function never charges a battery itself.
        """
        duration = _seconds(duration_seconds, 'duration_seconds')
        if duration == 0:
            raise ValueError('PV interval must have positive duration')
        fraction = _fraction(output_fraction)
        absorption = _number(absorption_limit_kw, 'absorption_limit_kw', True)
        available = self.available_power_kw(fraction)
        used = min(available, absorption)
        curtailed = available - used
        return PVStep(duration, fraction, absorption, available, used, curtailed,
                      energy_kwh(available, duration), energy_kwh(used, duration),
                      energy_kwh(curtailed, duration))
