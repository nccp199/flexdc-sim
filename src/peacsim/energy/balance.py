"""Actual facility power balance and a compact, contiguous energy ledger.

No device execution or action coordination: the caller binds component results
to the same elapsed interval before constructing a BalanceStep. Component
durations can be checked here, but do not independently prove their provenance.
"""
from dataclasses import dataclass, field
import math

from peacsim.energy.battery import BatteryStep
from peacsim.energy.config import DEFAULT_TOLERANCES
from peacsim.energy.contracts import EnergyInterval, energy_kwh
from peacsim.energy.cooling import CoolingStep
from peacsim.energy.renewables import PVStep
from peacsim.metrics.tracking import _number, _run_id, _seconds


BALANCE_MODEL_VERSION = 'flexdc-balance-v1'
_FLOW_KEYS = ('it', 'cooling', 'other', 'battery_charge', 'battery_discharge',
              'pv_available', 'pv_used', 'pv_curtailed', 'grid_import', 'battery_loss')


def _sides(values):
    supply = math.fsum(values[k] for k in ('grid_import', 'pv_used', 'battery_discharge'))
    demand = math.fsum(values[k] for k in ('it', 'cooling', 'other', 'battery_charge'))
    return supply, demand


def _residual(values):
    return math.fsum([values['grid_import'], values['pv_used'], values['battery_discharge'],
                      -values['it'], -values['cooling'], -values['other'], -values['battery_charge']])


def _require_close(matches, left, right, label):
    if not matches(left, right):
        raise ValueError(label + ' does not balance')


def _check_energy(values, before, after, matches):
    _require_close(matches, *_sides(values), 'facility energy')
    _require_close(matches, values['pv_available'],
                   math.fsum([values['pv_used'], values['pv_curtailed']]), 'PV energy')
    _require_close(matches, math.fsum([before, values['battery_charge']]),
                   math.fsum([after, values['battery_discharge'], values['battery_loss']]),
                   'battery energy')


@dataclass(frozen=True)
class BalanceStep:
    """Validated actual component outputs for one interval, with derived import.

    The sole default is the approved first-version assumption of no separately
    modeled other facility load. Missing device outputs never mean zero power.
    """

    interval: EnergyInterval
    cooling: CoolingStep
    pv: PVStep
    battery: BatteryStep
    other_power_kw: float = 0.
    grid_import_power_kw: float = field(init=False, default=0.)

    def __post_init__(self):
        for name, cls in [('interval', EnergyInterval), ('cooling', CoolingStep),
                          ('pv', PVStep), ('battery', BatteryStep)]:
            if not isinstance(getattr(self, name), cls):
                raise TypeError(name + ' must be a ' + cls.__name__)
        duration = self.interval.duration_seconds
        for name in ('cooling', 'pv', 'battery'):
            if _seconds(getattr(self, name).duration_seconds, name + ' duration') != duration:
                raise ValueError(name + ' duration differs from the IT interval')
        object.__setattr__(self, 'other_power_kw', _number(self.other_power_kw, 'other_power_kw', True))
        tol = DEFAULT_TOLERANCES
        cooling_it = _number(self.cooling.actual_it_power_kw, 'cooling IT power', True)
        _require_close(tol.power_matches, cooling_it, self.interval.actual_it_power_kw, 'cooling IT input')
        powers = self.powers_kw
        for name, value in powers.items():
            _number(value, name + ' power', True)
        if powers['battery_charge'] > 0 and powers['battery_discharge'] > 0:
            raise ValueError('cannot charge and discharge in the same interval')
        _require_close(tol.power_matches, powers['pv_available'],
                       math.fsum([powers['pv_used'], powers['pv_curtailed']]), 'PV power')
        for name in ('energy_before_kwh', 'energy_after_kwh', 'loss_energy_kwh', 'soc_after'):
            _number(getattr(self.battery, name), 'battery ' + name, True)
        if self.battery.soc_after > 1:
            raise ValueError('battery SOC must be in [0, 1]')

        net = -_residual(powers)  # grid_import is still zero at this point.
        if net < 0 and not tol.power_matches(*_sides(powers)):
            raise ValueError('actual supply exceeds demand; export is not allowed')
        # Only a tolerance-checked floating residual can reach this branch with
        # negative net. Its power and energy residuals remain visible below.
        object.__setattr__(self, 'grid_import_power_kw', max(0., net))
        energies = self.energies_kwh
        for key, value in [('cooling', self.cooling.cooling_energy_kwh),
                           ('pv_available', self.pv.pv_available_energy_kwh),
                           ('pv_used', self.pv.pv_used_energy_kwh),
                           ('pv_curtailed', self.pv.pv_curtailed_energy_kwh),
                           ('battery_charge', self.battery.charge_energy_kwh),
                           ('battery_discharge', self.battery.discharge_energy_kwh)]:
            _require_close(tol.energy_matches, _number(value, key + ' energy', True),
                           energies[key], key + ' power/time/energy')
        _require_close(tol.power_matches, *self._power_sides(), 'facility power')
        _check_energy(energies, self.battery.energy_before_kwh, self.battery.energy_after_kwh,
                      tol.energy_matches)

    def _power_sides(self):
        return _sides(self.powers_kw)

    @property
    def powers_kw(self):
        """Return a fresh mapping of actual interval-average powers."""
        return dict(it=self.interval.actual_it_power_kw, cooling=self.cooling.cooling_power_kw,
                    other=self.other_power_kw, battery_charge=self.battery.charge_power_kw,
                    battery_discharge=self.battery.discharge_power_kw,
                    pv_available=self.pv.pv_available_power_kw, pv_used=self.pv.pv_used_power_kw,
                    pv_curtailed=self.pv.pv_curtailed_power_kw, grid_import=self.grid_import_power_kw)

    @property
    def energies_kwh(self):
        values = {k: energy_kwh(v, self.interval.duration_seconds) for k, v in self.powers_kw.items()}
        values['battery_loss'] = self.battery.loss_energy_kwh
        return values

    @property
    def power_balance_residual_kw(self):
        return _residual(self.powers_kw)

    @property
    def energy_balance_residual_kwh(self):
        return _residual(self.energies_kwh)

    def to_dict(self):
        return dict(version=BALANCE_MODEL_VERSION, run_id=self.interval.run_id,
                    start_seconds=self.interval.start_seconds, end_seconds=self.interval.end_seconds,
                    duration_seconds=self.interval.duration_seconds, power_kw=self.powers_kw,
                    energy_kwh=self.energies_kwh,
                    battery=dict(energy_before_kwh=self.battery.energy_before_kwh,
                                 energy_after_kwh=self.battery.energy_after_kwh,
                                 soc_after=self.battery.soc_after),
                    power_balance_residual_kw=self.power_balance_residual_kw,
                    energy_balance_residual_kwh=self.energy_balance_residual_kwh)


class EnergyLedger:
    """One run/window, storing only totals, coverage and battery endpoints."""

    def __init__(self, run_id, start_seconds, end_seconds):
        self._run_id = _run_id(run_id)
        self._start = _seconds(start_seconds, 'window start')
        self._end = _seconds(end_seconds, 'window end')
        if self._end <= self._start:
            raise ValueError('ledger window must have positive duration')
        self._cursor = self._start
        self._count = 0
        self._totals = dict.fromkeys(_FLOW_KEYS, 0.)
        self._battery_initial = self._battery_final = self._battery_soc = None

    def append(self, step):
        """Check before committing; an invalid append changes no ledger state."""
        if not isinstance(step, BalanceStep):
            raise TypeError('step must be a BalanceStep')
        interval = step.interval
        if interval.run_id != self._run_id:
            raise ValueError('step belongs to a different run')
        if interval.start_seconds != self._cursor or interval.end_seconds > self._end:
            raise ValueError('step must cover the next contiguous part of the ledger window')
        tol = DEFAULT_TOLERANCES
        before, after = step.battery.energy_before_kwh, step.battery.energy_after_kwh
        if self._battery_final is not None:
            _require_close(tol.energy_matches, before, self._battery_final, 'battery continuity')
        initial = before if self._battery_initial is None else self._battery_initial
        energies = step.energies_kwh
        totals = {k: _number(math.fsum([self._totals[k], energies[k]]), k + ' total', True)
                  for k in _FLOW_KEYS}
        _check_energy(totals, initial, after, tol.cumulative_energy_matches)
        self._totals = totals
        self._battery_initial, self._battery_final = initial, after
        self._battery_soc = step.battery.soc_after
        self._cursor = interval.end_seconds
        self._count += 1

    def summary(self):
        """Return a copy; complete=False explicitly marks unfinished coverage."""
        return dict(version=BALANCE_MODEL_VERSION, run_id=self._run_id,
                    start_seconds=self._start, end_seconds=self._end,
                    covered_end_seconds=self._cursor, covered_duration_seconds=self._cursor-self._start,
                    complete=self._cursor == self._end, interval_count=self._count,
                    energy_kwh=dict(self._totals),
                    battery=dict(initial_energy_kwh=self._battery_initial,
                                 final_energy_kwh=self._battery_final, final_soc=self._battery_soc),
                    energy_balance_residual_kwh=_residual(self._totals))
