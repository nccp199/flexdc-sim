"""Optional boundary-aligned operating conditions for a scenario."""
from peacsim.metrics.tracking import _number, _seconds


class DynamicEvents:
    def __init__(self, spec, *, step, end, job_types):
        keys = {"price", "grid_limit", "task_priority"}
        if set(spec) != keys:
            raise ValueError("dynamic events require price, grid_limit and task_priority lists")
        self._events = {}
        for kind in sorted(keys):
            rows = []
            previous_end = 0
            value_key = {"price": "multiplier", "grid_limit": "kw", "task_priority": "type_id"}[kind]
            if not isinstance(spec[kind], list):
                raise ValueError(f"{kind} must be an event list")
            for event in spec[kind]:
                if set(event) != {"start", "end", value_key}:
                    raise ValueError(f"invalid {kind} event fields")
                start = _seconds(event["start"], "event start")
                stop = _seconds(event["end"], "event end")
                if not previous_end <= start < stop <= end or start % step or stop % step:
                    raise ValueError(f"{kind} events must be ordered, disjoint and aligned")
                value = (_seconds(event[value_key], "priority task type") if kind == "task_priority"
                         else _number(event[value_key], value_key, True))
                if kind == "task_priority" and value >= job_types:
                    raise ValueError("unknown priority task type")
                rows.append((start, stop, value))
                previous_end = stop
            self._events[kind] = tuple(rows)

    def _active(self, kind, t):
        return next((v for start, end, v in self._events[kind] if start <= t < end), None)

    def price_multiplier(self, t):
        value = self._active("price", t)
        return 1.0 if value is None else value

    def at(self, t):
        priority = self._active("task_priority", t)
        return dict(grid_limit_kw=self._active("grid_limit", t), priority_types=[] if priority is None else [priority])
