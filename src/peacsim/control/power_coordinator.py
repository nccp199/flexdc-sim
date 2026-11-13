"""Replaceable grid-to-IT algebra, not a battery policy or an optimizer.

PV is a current-value hold estimate. Device execution and actual metering still
belong to S4b; an IT envelope is not a promise of task-level reachability.
"""
from dataclasses import dataclass
import math

from peacsim.control.joint_action import JointAction
from peacsim.energy.battery import BatteryPowerLimits
from peacsim.energy.facility import BatteryRequest
from peacsim.metrics.tracking import _number


@dataclass(frozen=True)
class CoordinationPlan:
    action: JointAction
    requested_battery: BatteryRequest
    grid_target_kw: float
    pv_available_estimate_kw: float
    raw_it_budget_kw: float
    predicted_pv_used_kw: float
    predicted_grid_import_kw: float
    predicted_grid_error_kw: float
    limits: tuple


def coordinate_power(*, grid_target_kw, pv_available_kw, cooling_power_ratio,
                     it_min_power_kw, it_max_power_kw, battery_request, battery_limits):
    """Allocate the residual target to IT after one bounded battery request.

    The caller obtains battery_limits for the intended hold duration. Fixed
    cooling gives IT = (grid + PV + discharge - charge) / (1 + ratio).
    Clip to the coarse cluster envelope; retain the resulting target residual.
    """
    grid = _number(grid_target_kw, 'grid_target_kw', True)
    pv = _number(pv_available_kw, 'pv_available_kw', True)
    ratio = _number(cooling_power_ratio, 'cooling_power_ratio', True)
    low = _number(it_min_power_kw, 'it_min_power_kw', True)
    high = _number(it_max_power_kw, 'it_max_power_kw', True)
    if low > high:
        raise ValueError('IT minimum exceeds maximum')
    if not isinstance(battery_request, BatteryRequest) or not isinstance(battery_limits, BatteryPowerLimits):
        raise TypeError('expected BatteryRequest and BatteryPowerLimits')
    charge = min(battery_request.charge_power_kw,
                 _number(battery_limits.charge_power_kw, 'battery charge limit', True))
    discharge = min(battery_request.discharge_power_kw,
                    _number(battery_limits.discharge_power_kw, 'battery discharge limit', True))
    limits = []
    if charge < battery_request.charge_power_kw or discharge < battery_request.discharge_power_kw:
        limits.append('battery_power_or_soc_limit')
    raw = _number(math.fsum([grid, pv, discharge, -charge]) / (1. + ratio), 'raw IT budget')
    it = min(high, max(low, raw))
    if it != raw:
        limits.append('cluster_power_envelope')
    load = _number(it * (1. + ratio), 'estimated facility load', True)
    allowed_discharge = min(discharge, max(0., load - pv))
    if allowed_discharge < discharge:
        limits.append('pv_first_no_export')
    discharge = allowed_discharge
    absorption = max(0., math.fsum([load, charge, -discharge]))
    used = min(pv, absorption)
    predicted = max(0., math.fsum([load, charge, -discharge, -used]))
    action = JointAction(it, BatteryRequest(charge, discharge))
    return CoordinationPlan(action, battery_request, grid, pv, raw, used,
                            predicted, predicted - grid, tuple(limits))
