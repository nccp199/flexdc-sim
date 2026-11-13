"""S4b wiring: PV-first execution and metering, using the existing S3 models."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from peacsim.energy.balance import BalanceStep, EnergyLedger
from peacsim.energy.battery import Battery
from peacsim.energy.cooling import Cooling
from peacsim.energy.contracts import EnergyInterval
from peacsim.energy.renewables import PV
from peacsim.metrics.tracking import _number
from peacsim.scenarios import ScenarioBundle


@dataclass(frozen=True)
class BatteryRequest:
    """Facility-side kW request, supplied before the interval is advanced."""
    charge_power_kw: float = 0.
    discharge_power_kw: float = 0.

    def __post_init__(self):
        for name in ('charge_power_kw', 'discharge_power_kw'):
            object.__setattr__(self, name, _number(getattr(self, name), name, True))
        if self.charge_power_kw > 0 and self.discharge_power_kw > 0:
            raise ValueError('cannot request charging and discharging together')


class FacilityEnergy:
    """Own independent devices and one ledger for a declared run/window.

    advance consumes a real IT interval with a previously selected request.
    It is also usable for fixed-trajectory checks; it does not select actions.
    No task mutation, IT target conversion, terminal correction or reward.
    """
    def __init__(self, scenario, run_id):
        if not isinstance(scenario, ScenarioBundle):
            raise TypeError('scenario must be a ScenarioBundle')
        self._scenario = scenario
        self._battery = Battery(scenario.environment.battery)
        self._pv = PV(scenario.environment.pv)
        self._cooling = Cooling(scenario.environment.cooling)
        self._ledger = EnergyLedger(run_id, 0, scenario.observation_end_seconds)
        self._currency = scenario.metadata['signals']['price_per_kwh']['unit'][:-4]
        self._cost = self._carbon = 0.
        self._last_rows = ()
        self._failed = False

    def advance(self, interval, request):
        if self._failed:
            raise RuntimeError('failed facility run; create fresh devices and ledger')
        if not isinstance(interval, EnergyInterval) or not isinstance(request, BatteryRequest):
            raise TypeError('expected EnergyInterval and BatteryRequest')
        # Bind to the existing ledger's declared coverage before executing devices;
        # the ledger remains responsible for its power/energy/continuity checks.
        cover = self._ledger.summary()
        if (interval.run_id != cover['run_id']
                or interval.start_seconds != cover['covered_end_seconds']
                or interval.end_seconds > cover['end_seconds']):
            raise ValueError('IT interval must match this ledger and its next covered interval')
        pieces = self._scenario.segments(interval.start_seconds, interval.end_seconds)
        rows = []
        try:
            for signal in pieces:
                a, b = signal['start_seconds'], signal['end_seconds']
                part = EnergyInterval(interval.run_id, a, b, interval.actual_it_power_kw)
                cooling = self._cooling.step(b-a, actual_it_power_kw=part.actual_it_power_kw)
                load = math.fsum([part.actual_it_power_kw, cooling.cooling_power_kw])
                pv_available = self._pv.available_power_kw(signal['pv_fraction'])
                # PV first. Charging is never invented by the meter. The battery
                # applies its own power/SOC limits to this facility-safe request.
                allowed_discharge = min(request.discharge_power_kw, max(0., load-pv_available))
                battery = self._battery.step(b-a, charge_power_kw=request.charge_power_kw,
                                             discharge_power_kw=allowed_discharge)
                pv = self._pv.step(b-a, output_fraction=signal['pv_fraction'],
                                   absorption_limit_kw=max(0., math.fsum([
                                       load, battery.charge_power_kw, -battery.discharge_power_kw])))
                balance = BalanceStep(part, cooling, pv, battery)
                grid_energy = balance.energies_kwh['grid_import']
                cost = _number(grid_energy * signal['price_per_kwh'], 'grid purchase cost')
                carbon = _number(grid_energy * signal['carbon_kg_per_kwh'], 'grid carbon', True)
                total_cost = _number(math.fsum([self._cost, cost]), 'total grid cost')
                total_carbon = _number(math.fsum([self._carbon, carbon]), 'total grid carbon', True)
                self._ledger.append(balance)
                self._cost, self._carbon = total_cost, total_carbon
                reasons = []
                if allowed_discharge < request.discharge_power_kw:
                    reasons.append('pv_first_no_export')
                if (battery.charge_power_kw < request.charge_power_kw
                        or battery.discharge_power_kw < allowed_discharge):
                    reasons.append('battery_power_or_soc_limit')
                row = balance.to_dict()
                row.update(request=dict(issued_at_seconds=interval.start_seconds, **asdict(request)),
                           facility_request=dict(charge_power_kw=request.charge_power_kw,
                                                 discharge_power_kw=allowed_discharge),
                           execution_limits=reasons, signals=dict(signal),
                           grid_purchase_cost=dict(amount=cost, currency=self._currency),
                           grid_carbon_kg=carbon)
                rows.append(row)
        except Exception:
            # Match the existing simulator's fail/reset contract. No automatic
            # retry or rollback of a partially executed physical interval.
            self._failed = True
            raise
        self._last_rows = tuple(rows)
        return self.last_rows()

    def last_rows(self):
        return deepcopy(self._last_rows)

    def battery_power_limits(self, duration_seconds):
        """Forward the existing read-only BESS limit query to a coordinator."""
        return self._battery.power_limits(duration_seconds)

    def summary(self):
        result = self._ledger.summary()
        result.update(valid=not self._failed,
                      grid_purchase_cost=dict(amount=self._cost, currency=self._currency),
                      grid_carbon_kg=self._carbon,
                      carbon_model='external average intensity; not marginal emissions',
                      battery_current=dict(energy_kwh=self._battery.energy_kwh, soc=self._battery.soc))
        return result
