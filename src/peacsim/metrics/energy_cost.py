"""S2c: observed IT energy and an explicitly priced, constant-tariff estimate.

Pure calculation in W, seconds and kWh. No contract charges, performance
penalties, rewards, grid-purchase assumptions, file access or simulation RNG.
"""
from dataclasses import asdict, dataclass
import math

from peacsim.metrics.tracking import PowerInterval, _number, _run_id, _seconds


ENERGY_COST_VERSION = 'flexdc-it-energy-cost-v1'


@dataclass(frozen=True)
class ElectricityTariff:
    """One explicit fixed price; currency and source are caller-supplied labels.

    Zero and negative prices are allowed. No currency conversion is performed.
    Time-varying tariffs belong to the later scenario integration stage.
    """
    price_per_kwh: float
    currency: str
    source: str

    def __post_init__(self):
        object.__setattr__(self, 'price_per_kwh', _number(self.price_per_kwh, 'price_per_kwh'))
        for name in ('currency', 'source'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(name + ' must be a nonempty string')
            object.__setattr__(self, name, value.strip())


def evaluate_it_energy_cost(intervals, *, run_id, window_start_seconds, window_end_seconds, tariff=None):
    """Evaluate exactly one covered window, allowing an empty zero-duration run.

    Input intervals carry the same identity and elapsed power used by S2a/S2b.
    Equal powers are grouped before summation so equivalent partitions have
    identical results. No time is filled, rounded to hours, or inferred from P/R.
    """
    _run_id(run_id)
    start = _seconds(window_start_seconds, 'window start')
    end = _seconds(window_end_seconds, 'window end')
    if end < start:
        raise ValueError('window end must not precede window start')
    if tariff is not None and not isinstance(tariff, ElectricityTariff):
        raise TypeError('tariff must be an ElectricityTariff or None')
    cursor, count, durations_by_power = start, 0, {}
    for interval in intervals:
        if not isinstance(interval, PowerInterval) or interval.run_id != run_id:
            raise ValueError('each power interval must belong to this run')
        if interval.start_seconds != cursor or interval.end_seconds > end:
            raise ValueError('power intervals must cover the requested window in order without gaps or overlaps')
        power = interval.actual_it_power_w
        durations_by_power[power] = durations_by_power.get(power, 0) + interval.end_seconds - interval.start_seconds
        cursor = interval.end_seconds
        count += 1
    if cursor != end:
        raise ValueError('incomplete power interval coverage')
    # W -> kW and seconds -> hours, each converted exactly once.
    energy = _number(math.fsum((power / 1000) * (seconds / 3600)
                              for power, seconds in sorted(durations_by_power.items())), 'it_energy_kwh', True)
    amount = (_number(energy * tariff.price_per_kwh, 'estimated electricity cost')
              if tariff is not None else None)
    return dict(schema_version=ENERGY_COST_VERSION, run_id=run_id, status='complete',
                power_boundary='IT', energy_unit='kWh', power_unit='W', time_unit='s',
                window_start_seconds=start, window_end_seconds=end,
                covered_duration_seconds=end - start, source_interval_count=count,
                it_energy_kwh=energy,
                it_electricity_cost_estimate=dict(
                    status='estimated' if tariff is not None else 'not_configured',
                    amount=amount, currency=tariff.currency if tariff is not None else None,
                    model='constant_tariff', tariff=asdict(tariff) if tariff is not None else None),
                grid_purchase_cost=dict(status='not_connected', amount=None),
                battery_degradation_cost=dict(status='not_connected', amount=None))
