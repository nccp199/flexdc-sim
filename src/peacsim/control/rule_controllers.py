"""S4d deterministic battery rules; no device execution or future data input."""
from dataclasses import dataclass

from peacsim.energy.facility import BatteryRequest
from peacsim.metrics.tracking import _number


RULE_VERSION = 'flexdc-battery-rules-v1'
RULE_NAMES = ('battery_standby', 'pv_surplus_grid_support')


@dataclass(frozen=True)
class RuleDecision:
    rule: str
    battery_request: BatteryRequest
    reason: str


def decide_battery(rule, *, estimated_facility_load_kw, pv_available_kw, grid_target_kw):
    """Select an unclipped request; S4c/S4b retain all existing device limits.

    Participating rule: charge estimated PV surplus, otherwise discharge only
    the estimated import above target, otherwise stand by. No grid-price
    arbitrage, target filling with grid charging, SOC restoration or tuning.
    """
    if rule not in RULE_NAMES:
        raise ValueError('unknown battery rule: ' + str(rule))
    load = _number(estimated_facility_load_kw, 'estimated_facility_load_kw', True)
    pv = _number(pv_available_kw, 'pv_available_kw', True)
    target = _number(grid_target_kw, 'grid_target_kw', True)
    if rule == 'battery_standby':
        return RuleDecision(rule, BatteryRequest(), 'battery_standby')
    if pv > load:
        return RuleDecision(rule, BatteryRequest(charge_power_kw=pv-load), 'estimated_pv_surplus')
    excess = load - pv - target
    if excess > 0:
        return RuleDecision(rule, BatteryRequest(discharge_power_kw=excess), 'estimated_import_above_target')
    return RuleDecision(rule, BatteryRequest(), 'no_pv_surplus_or_import_excess')
