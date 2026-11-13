"""One IT budget and one facility-side battery request, both in kW."""
from dataclasses import dataclass

from peacsim.energy.facility import BatteryRequest
from peacsim.metrics.tracking import _number


@dataclass(frozen=True)
class JointAction:
    it_power_budget_kw: float
    battery_request: BatteryRequest = BatteryRequest()

    def __post_init__(self):
        budget = _number(self.it_power_budget_kw, 'it_power_budget_kw', True)
        _number(budget * 1000., 'IT budget converted to W', True)
        object.__setattr__(self, 'it_power_budget_kw', budget)
        if not isinstance(self.battery_request, BatteryRequest):
            raise TypeError('battery_request must be a BatteryRequest')
