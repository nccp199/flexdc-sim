"""Minimal BESS: one energy state, fixed efficiencies and power/SOC limits.

The owner chooses when to charge/discharge and calls step once per elapsed
interval. No prices, rewards, terminal correction, thermal or ageing model.
"""
from dataclasses import dataclass
import math

from peacsim.energy.config import BatteryConfig
from peacsim.energy.contracts import SECONDS_PER_HOUR, energy_kwh
from peacsim.metrics.tracking import _number, _seconds


BATTERY_MODEL_VERSION = 'flexdc-battery-v1'


@dataclass(frozen=True)
class BatteryPowerLimits:
    charge_power_kw: float
    discharge_power_kw: float


@dataclass(frozen=True)
class BatteryStep:
    duration_seconds: int
    requested_charge_power_kw: float
    requested_discharge_power_kw: float
    charge_power_kw: float
    discharge_power_kw: float
    charge_energy_kwh: float
    discharge_energy_kwh: float
    loss_energy_kwh: float
    energy_before_kwh: float
    energy_after_kwh: float
    soc_after: float


class Battery:
    """Stateful single battery; SOC is derived from the stored energy only."""

    def __init__(self, config):
        if not isinstance(config, BatteryConfig):
            raise TypeError('config must be a BatteryConfig')
        self._config = config
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def energy_kwh(self):
        return self._energy_kwh

    @property
    def soc(self):
        return self._energy_kwh / self._config.capacity_kwh

    def reset(self):
        """Start a new episode at its configured initial SOC, not at settlement."""
        self._energy_kwh = self._config.capacity_kwh * self._config.initial_soc

    def power_limits(self, duration_seconds):
        """Read-only limits on interval-average facility-side power.

        A later coordinator may choose a smaller final request, e.g. to avoid
        export. Querying these limits never charges or discharges the battery.
        """
        duration = _seconds(duration_seconds, 'duration_seconds')
        if duration == 0:
            raise ValueError('battery interval must have positive duration')
        hours = duration / SECONDS_PER_HOUR
        cfg = self._config
        room = max(0., cfg.capacity_kwh * cfg.soc_max - self._energy_kwh)
        available = max(0., self._energy_kwh - cfg.capacity_kwh * cfg.soc_min)
        return BatteryPowerLimits(
            min(cfg.max_charge_power_kw, room / hours / cfg.charge_efficiency),
            min(cfg.max_discharge_power_kw, available / hours * cfg.discharge_efficiency),
        )

    def step(self, duration_seconds, *, charge_power_kw=0., discharge_power_kw=0.):
        """Limit a request, account for the executed power, then update energy.

        Both inputs are nonnegative requests in kW; at most one may be positive.
        If a boundary is reached inside this interval, the returned power is the
        executable average for the whole interval. Call again each physical step
        when a control request is held across several steps.
        """
        requested_charge = _number(charge_power_kw, 'charge_power_kw', True)
        requested_discharge = _number(discharge_power_kw, 'discharge_power_kw', True)
        if requested_charge > 0 and requested_discharge > 0:
            raise ValueError('cannot charge and discharge in the same interval')
        duration = _seconds(duration_seconds, 'duration_seconds')
        limits = self.power_limits(duration)
        charge = min(requested_charge, limits.charge_power_kw)
        discharge = min(requested_discharge, limits.discharge_power_kw)
        charge_energy = energy_kwh(charge, duration)
        discharge_energy = energy_kwh(discharge, duration)
        cfg, before = self._config, self._energy_kwh
        stored = charge_energy * cfg.charge_efficiency
        withdrawn = discharge_energy / cfg.discharge_efficiency
        after = math.fsum([before, stored, -withdrawn])
        # Power was limited before integration; only floating-point roundoff can
        # cross a bound here. Never use an unclipped request to debit the state.
        after = min(cfg.capacity_kwh * cfg.soc_max,
                    max(cfg.capacity_kwh * cfg.soc_min, after))
        loss = math.fsum([charge_energy * (1 - cfg.charge_efficiency),
                         withdrawn * (1 - cfg.discharge_efficiency)])
        result = BatteryStep(duration, requested_charge, requested_discharge,
                             charge, discharge, charge_energy, discharge_energy,
                             loss, before, after, after / cfg.capacity_kwh)
        self._energy_kwh = after
        return result
