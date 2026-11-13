"""S3a unit boundary; reuse S2 interval semantics without changing the engine.

All powers here are actual, nonnegative interval-average kW. Native intervals
are left-closed/right-open integer seconds. Requests and endpoint samples must
not substitute for elapsed power. These functions do not collect a history or
settle a bill; coverage/order remain the responsibility of the owning ledger.
"""
from dataclasses import dataclass

from peacsim.metrics.tracking import (
    PowerInterval, _number, _run_id, _seconds, interval_from_transition,
)


ENERGY_CONTRACT_VERSION = 'flexdc-energy-contract-v1'
WATTS_PER_KW = 1000
SECONDS_PER_HOUR = 3600


def energy_kwh(power_kw, duration_seconds):
    """Integrate one constant/average nonnegative power over actual seconds.

    Zero duration has zero energy. A changing power must first be split at its
    changes or supplied as an interval average, not an endpoint measurement.
    """
    power = _number(power_kw, 'power_kw', True)
    duration = _seconds(duration_seconds, 'duration_seconds')
    return _number(power * (duration / SECONDS_PER_HOUR), 'energy_kwh', True)


@dataclass(frozen=True)
class EnergyInterval:
    """Owned scalar energy input; no device values defaulted to fictitious zero."""

    run_id: str
    start_seconds: int
    end_seconds: int
    actual_it_power_kw: float

    def __post_init__(self):
        _run_id(self.run_id)
        for name in ('start_seconds', 'end_seconds'):
            object.__setattr__(self, name, _seconds(getattr(self, name), name))
        if self.end_seconds <= self.start_seconds:
            raise ValueError('energy interval must have positive duration')
        object.__setattr__(self, 'actual_it_power_kw',
                           _number(self.actual_it_power_kw, 'actual_it_power_kw', True))

    @property
    def duration_seconds(self):
        return self.end_seconds - self.start_seconds

    @property
    def actual_it_energy_kwh(self):
        return energy_kwh(self.actual_it_power_kw, self.duration_seconds)


def energy_interval_from_power_interval(interval):
    """The sole W -> kW entry to this layer, from an existing S2 interval.

    Requiring PowerInterval also prevents accidentally converting an existing
    EnergyInterval a second time. S2c keeps its independent native W calculation.
    """
    if not isinstance(interval, PowerInterval):
        raise TypeError('expected a native PowerInterval in W')
    return EnergyInterval(interval.run_id, interval.start_seconds, interval.end_seconds,
                          interval.actual_it_power_w / WATTS_PER_KW)


def energy_interval_from_transition(record, *, run_id, expected_episode_id):
    """Reuse S2 projection, including initialization None and episode checks.

    Call once per successful advance_cycle. Re-reading a record is a pure
    conversion, not a new elapsed interval; this adapter does not deduplicate.
    """
    interval = interval_from_transition(record, run_id=run_id,
                                        expected_episode_id=expected_episode_id)
    return None if interval is None else energy_interval_from_power_interval(interval)
