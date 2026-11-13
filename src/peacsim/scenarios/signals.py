"""Finite piecewise-constant signals; integer seconds, no implicit extrapolation."""
from dataclasses import dataclass

from peacsim.metrics.tracking import _number, _seconds


@dataclass(frozen=True)
class Signal:
    start_seconds: int
    interval_seconds: int
    values: tuple
    unit: str
    source: str

    def __post_init__(self):
        # A mapped source can begin before scenario t=0 (e.g. a PV day).
        origin = _number(self.start_seconds, 'signal origin')
        magnitude = _seconds(abs(origin), 'signal origin magnitude')
        object.__setattr__(self, 'start_seconds', -magnitude if origin < 0 else magnitude)
        object.__setattr__(self, 'interval_seconds', _seconds(self.interval_seconds, 'signal interval'))
        object.__setattr__(self, 'values', tuple(_number(v, 'signal value') for v in self.values))
        if self.interval_seconds == 0 or not self.values:
            raise ValueError('signal needs a positive interval and nonempty values')
        for name in ('unit', 'source'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError('signal ' + name + ' must be explicit')

    @property
    def end_seconds(self):
        return self.start_seconds + self.interval_seconds * len(self.values)

    def at(self, seconds):
        seconds = _seconds(seconds, 'signal time')
        if not self.start_seconds <= seconds < self.end_seconds:
            raise ValueError('signal time is outside its half-open coverage')
        return self.values[(seconds - self.start_seconds) // self.interval_seconds]

    def segments(self, start, end):
        start, end = _seconds(start, 'segment start'), _seconds(end, 'segment end')
        if not self.start_seconds <= start < end <= self.end_seconds:
            raise ValueError('signal does not cover the requested positive interval')
        result = []
        while start < end:
            index = (start - self.start_seconds) // self.interval_seconds
            stop = min(end, self.start_seconds + (index + 1) * self.interval_seconds)
            result.append((start, stop, self.values[index]))
            start = stop
        return tuple(result)
