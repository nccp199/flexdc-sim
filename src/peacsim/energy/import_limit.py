"""Shared real-time protection for a binding import limit."""
from dataclasses import asdict

from .facility import BatteryRequest


def protect_import(*, limit, it_kw, cooling_ratio, pv_kw, request, battery_limits):
    """Reduce charging, then add discharge and finally curtail reducible IT."""
    factor = 1 + cooling_ratio
    net_kw = it_kw * factor - pv_kw
    charge = min(request.charge_power_kw, battery_limits.charge_power_kw)
    discharge = min(request.discharge_power_kw, battery_limits.discharge_power_kw, max(0., net_kw))
    raw_grid = max(0., net_kw + charge - discharge)
    allowed_it = it_kw
    corrected = request
    if raw_grid > limit + 1e-9:
        charge = min(charge, max(0., limit - net_kw))
        discharge = min(battery_limits.discharge_power_kw, max(discharge, net_kw - limit))
        corrected = BatteryRequest(charge, discharge)
        allowed_it = max(0., min(it_kw, (limit + pv_kw + discharge - charge) / factor))
    return corrected, allowed_it, dict(
        grid_limit_kw=limit, unprotected_it_kw=it_kw, unprotected_grid_kw=raw_grid,
        it_safety_target_kw=allowed_it, requested_battery=asdict(request),
        protected_battery=asdict(corrected),
    )
