"""S2a native IT tracking: integer-second, left-closed/right-open intervals.

Pure evaluation only: no simulator access, I/O, RNG, implicit clipping, or legacy
report replacement. Targets are already resolved W values, held between updates.
The caller binds a unique run_id to BOTH the intervals and the target schedule.
"""
from dataclasses import dataclass
import math
from numbers import Real


METRIC_VERSION = 'flexdc-tracking-v1'
TRANSITION_VERSION = 'flexdc-transition-v1'
P90_METHOD = 'inverse_duration_cdf_no_interpolation'


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + ' must be an integer number of seconds')
    if not math.isfinite(value) or value < 0 or value > 2**53 or int(value) != value:
        raise ValueError(name + ' must be an exact nonnegative integer second <= 2**53')
    return int(value)


def _number(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + ' must be a finite number')
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(name + ' must be finite' + (' and nonnegative' if nonnegative else ''))
    return result


def _run_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('run_id must be a nonempty string identifying one run')
    return value


@dataclass(frozen=True)
class PowerInterval:
    run_id: str
    start_seconds: int
    end_seconds: int
    actual_it_power_w: float

    def __post_init__(self):
        _run_id(self.run_id)
        for name in ('start_seconds', 'end_seconds'):
            object.__setattr__(self, name, _seconds(getattr(self, name), name))
        if self.end_seconds <= self.start_seconds:
            raise ValueError('power interval must have positive duration')
        object.__setattr__(self, 'actual_it_power_w',
                           _number(self.actual_it_power_w, 'actual_it_power_w', True))


@dataclass(frozen=True)
class TargetSchedule:
    run_id: str
    power_w: tuple
    update_interval_seconds: int
    valid_start_seconds: int
    valid_end_seconds: int
    reserve_w: float

    def __post_init__(self):
        _run_id(self.run_id)
        for name in ('update_interval_seconds', 'valid_start_seconds', 'valid_end_seconds'):
            object.__setattr__(self, name, _seconds(getattr(self, name), name))
        step = self.update_interval_seconds
        duration = self.valid_end_seconds - self.valid_start_seconds
        if step <= 0 or duration <= 0:
            raise ValueError('target update interval and valid duration must be positive')
        object.__setattr__(self, 'reserve_w', _number(self.reserve_w, 'reserve_w', True))
        # Copy the source; terminal native sample is permitted but adds no time.
        values = tuple(_number(v, 'target_power_w') for v in self.power_w)
        needed = (duration + step - 1) // step
        allowed = {needed, needed + 1} if duration % step == 0 else {needed}
        if len(values) not in allowed:
            raise ValueError('target samples do not match the declared validity window')
        object.__setattr__(self, 'power_w', values)


def interval_from_transition(record, *, run_id, expected_episode_id):
    """Project one successful S1 record into a compact immutable interval.

    A verified t=0 initialization returns None. NaNs in unrelated node progress
    or absent commands are legitimate and are NOT scanned. For a plain Simulator
    whose episode_id is None, the collector must supply a unique run_id and keep
    records bound to that one instance. This function cannot infer their origin.
    """
    _run_id(run_id)
    if not isinstance(record, dict):
        raise ValueError('a successful transition record is required')
    try:
        meta, interval = record['meta'], record['interval']
        if meta['schema_version'] != TRANSITION_VERSION:
            raise ValueError('unsupported transition schema')
        if meta['episode_id'] != expected_episode_id:
            raise ValueError('transition belongs to a different episode')
        start = _seconds(interval['start_time_seconds'], 'interval start')
        end = _seconds(interval['end_time_seconds'], 'interval end')
        duration = _seconds(interval['duration_seconds'], 'interval duration')
        time = _seconds(meta['time_seconds'], 'record time')
        index = _seconds(meta['sample_index'], 'sample index')
        if time != end or end - start != duration:
            raise ValueError('inconsistent transition times')
        if meta['kind'] == 'initialization':
            if (start != 0 or end != 0 or index != 0 or meta['previous_sample_index'] is not None
                    or interval['has_elapsed_time'] is not False
                    or interval['actual_it_power_w'] is not None):
                raise ValueError('invalid initialization record')
            return None
        if (meta['kind'] != 'interval' or interval['has_elapsed_time'] is not True or index == 0
                or _seconds(meta['previous_sample_index'], 'previous sample index') != index - 1):
            raise ValueError('invalid elapsed transition identity')
        return PowerInterval(run_id, start, end, interval['actual_it_power_w'])
    except (KeyError, TypeError) as exc:
        raise ValueError('missing or malformed transition fields') from exc


def evaluate_tracking(intervals, targets, *, window_start_seconds, window_end_seconds):
    """Evaluate exactly the requested window; reject gaps/overlaps/extra data.

    Returns JSON-compatible summary and split segment detail (W and seconds).
    PowerIntervals must be ordered and cover this window exactly. Explicitly
    prepare a subset to evaluate another window; no silent cropping is done.
    """
    if not isinstance(targets, TargetSchedule):
        raise ValueError('targets must be a TargetSchedule')
    start = _seconds(window_start_seconds, 'window start')
    end = _seconds(window_end_seconds, 'window end')
    if end <= start:
        raise ValueError('evaluation window must have positive duration')
    if start < targets.valid_start_seconds or end > targets.valid_end_seconds:
        raise ValueError('target validity does not cover evaluation window')
    cursor, segments, durations_by_error = start, [], {}
    count = 0
    for index, interval in enumerate(intervals):
        if not isinstance(interval, PowerInterval):
            raise ValueError('each input must be a PowerInterval')
        if interval.run_id != targets.run_id:
            raise ValueError('interval and target run_id differ')
        if interval.start_seconds != cursor or interval.end_seconds > end:
            raise ValueError('interval {}: expected start {}, got [{},{}); gap, overlap, order or window mismatch'
                             .format(index, cursor, interval.start_seconds, interval.end_seconds))
        while cursor < interval.end_seconds:
            target_index = (cursor - targets.valid_start_seconds) // targets.update_interval_seconds
            stop = min(interval.end_seconds, targets.valid_start_seconds +
                       (target_index + 1) * targets.update_interval_seconds)
            target = targets.power_w[target_index]
            error = _number(interval.actual_it_power_w - target, 'computed error_w')
            absolute = abs(error)
            normalized = _number(error / targets.reserve_w, 'normalized error') if targets.reserve_w > 0 else None
            duration = stop - cursor
            segments.append(dict(source_interval_index=index, target_index=target_index,
                                 start_seconds=cursor, end_seconds=stop, duration_seconds=duration,
                                 actual_it_power_w=interval.actual_it_power_w, target_power_w=target,
                                 error_w=error, abs_error_w=absolute, normalized_error=normalized))
            durations_by_error[absolute] = durations_by_error.get(absolute, 0) + duration
            cursor = stop
        count += 1
    if cursor != end or not count:
        raise ValueError('incomplete evaluation coverage: reached {}, required {}'.format(cursor, end))
    total = end - start
    ordered = sorted(durations_by_error.items())
    maximum = ordered[-1][0]
    # Group equal errors BEFORE summing: partitioning a constant segment cannot
    # change floating summation order/weights. Scaling avoids squaring overflow.
    if maximum:
        mean_ratio = math.fsum((e / maximum) * (d / total) for e, d in ordered)
        square_ratio = math.fsum((e / maximum)**2 * (d / total) for e, d in ordered)
        mae = maximum * min(1.0, mean_ratio)
        rmse = maximum * math.sqrt(min(1.0, square_ratio))
    else:
        mae = rmse = 0.0
    cumulative = 0
    for error, duration in ordered:
        cumulative += duration
        if 10 * cumulative >= 9 * total:
            p90 = error
            break
    metrics = dict(mae_w=mae, rmse_w=rmse, p90_abs_error_w=p90)
    for name, value in list(metrics.items()):
        metrics['normalized_' + name.removesuffix('_w')] = (
            _number(value / targets.reserve_w, 'normalized metric') if targets.reserve_w > 0 else None)
    return dict(metric_version=METRIC_VERSION, run_id=targets.run_id, power_boundary='IT',
                time_unit='s', power_unit='W', p90_method=P90_METHOD,
                window_start_seconds=start, window_end_seconds=end, covered_duration_seconds=total,
                source_interval_count=count, segment_count=len(segments), reserve_w=targets.reserve_w,
                normalization_status='applicable' if targets.reserve_w > 0 else 'not_applicable_zero_reserve',
                metrics=metrics, segments=segments)
