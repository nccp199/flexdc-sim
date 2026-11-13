"""Passive cooling electricity proportional to actual modeled IT power.

Assume sufficient cooling capacity and constant efficiency. IT is a heat-load
proxy; this component adds only facility cooling, not server fans or other IT.
There is no temperature state, weather input, controller or accumulated ledger.
"""
from dataclasses import dataclass

from peacsim.energy.config import CoolingConfig
from peacsim.energy.contracts import energy_kwh
from peacsim.metrics.tracking import _number, _seconds


COOLING_MODEL_VERSION = 'flexdc-cooling-v1'


@dataclass(frozen=True)
class CoolingStep:
    duration_seconds: int
    actual_it_power_kw: float
    cooling_power_kw: float
    cooling_energy_kwh: float


class Cooling:
    """Stateless fixed-ratio calculation with an immutable configuration."""

    def __init__(self, config):
        if not isinstance(config, CoolingConfig):
            raise TypeError('config must be a CoolingConfig')
        self._config = config

    @property
    def config(self):
        return self._config

    def power_kw(self, actual_it_power_kw):
        """Calculate facility cooling power, excluding the IT input itself."""
        it_power = _number(actual_it_power_kw, 'actual_it_power_kw', True)
        return _number(self._config.power_ratio * it_power, 'cooling_power_kw', True)

    def step(self, duration_seconds, *, actual_it_power_kw):
        """Integrate over one positive interval using actual average IT power.

        With a fixed ratio this is equivalent to integrating changing IT power
        in pieces. A request or endpoint sample is not an interval average.
        """
        duration = _seconds(duration_seconds, 'duration_seconds')
        if duration == 0:
            raise ValueError('cooling interval must have positive duration')
        it_power = _number(actual_it_power_kw, 'actual_it_power_kw', True)
        power = self.power_kw(it_power)
        return CoolingStep(duration, it_power, power, energy_kwh(power, duration))
