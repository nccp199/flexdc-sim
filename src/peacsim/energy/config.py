"""Energy comparison policy and configurable battery/PV/cooling parameters.

These are float64 bookkeeping tolerances, not sensor accuracy, physical slack,
or permission to clip invalid inputs. Compare both sides of the same equation:
abs(a-b) <= max(absolute_tolerance, relative * max(abs(a), abs(b))).
"""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

from peacsim.metrics.tracking import _number


@dataclass(frozen=True)
class EnergyTolerances:
    power_abs_kw: float = 1e-9
    energy_abs_kwh: float = 1e-12
    cumulative_energy_abs_kwh: float = 1e-9
    relative: float = 1e-10

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name,
                               _number(getattr(self, field.name), field.name, True))
        if self.relative >= 1:
            raise ValueError('relative tolerance must be less than 1')

    def _matches(self, actual, expected, absolute):
        return math.isclose(_number(actual, 'actual'), _number(expected, 'expected'),
                            abs_tol=absolute, rel_tol=self.relative)

    def power_matches(self, actual_kw, expected_kw):
        return self._matches(actual_kw, expected_kw, self.power_abs_kw)

    def energy_matches(self, actual_kwh, expected_kwh):
        return self._matches(actual_kwh, expected_kwh, self.energy_abs_kwh)

    def cumulative_energy_matches(self, actual_kwh, expected_kwh):
        return self._matches(actual_kwh, expected_kwh, self.cumulative_energy_abs_kwh)


DEFAULT_TOLERANCES = EnergyTolerances()


@dataclass(frozen=True)
class BatteryConfig:
    """Constant aggregate BESS parameters; powers are on the facility side.

    All values are explicit. Development presets live in configs/energy, not in
    the dynamics, and can be replaced when the complete scenario is calibrated.
    """

    capacity_kwh: float
    max_charge_power_kw: float
    max_discharge_power_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    soc_min: float
    soc_max: float
    initial_soc: float

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name,
                               _number(getattr(self, field.name), field.name, True))
        if self.capacity_kwh <= 0:
            raise ValueError('capacity_kwh must be positive')
        if not (0 < self.charge_efficiency <= 1 and 0 < self.discharge_efficiency <= 1):
            raise ValueError('charge and discharge efficiencies must be in (0, 1]')
        if not (0 <= self.soc_min < self.soc_max <= 1):
            raise ValueError('SOC bounds must satisfy 0 <= soc_min < soc_max <= 1')
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError('initial_soc must be inside the SOC bounds')


def load_battery_config(path):
    """Read the explicit battery section of a JSON scenario configuration."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return BatteryConfig(**data['battery'])


@dataclass(frozen=True)
class PVConfig:
    """Effective facility-side peak power; conversion losses are in the profile."""

    peak_power_kw: float

    def __post_init__(self):
        object.__setattr__(self, 'peak_power_kw', _number(self.peak_power_kw, 'peak_power_kw', True))


def load_pv_config(path):
    """Read the explicit PV section of a JSON scenario configuration."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return PVConfig(**data['pv'])


@dataclass(frozen=True)
class CoolingConfig:
    """Fixed ratio of facility cooling electricity to modeled IT electricity."""

    power_ratio: float

    def __post_init__(self):
        object.__setattr__(self, 'power_ratio', _number(self.power_ratio, 'power_ratio', True))


def load_cooling_config(path):
    """Read the explicit cooling section of a JSON scenario configuration."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return CoolingConfig(**data['cooling'])
