"""Validated, normalized task-type weights for explicitly selected control calls."""
import math
from collections.abc import Mapping

from peacsim.metrics.tracking import _number


def normalize_task_weights(values, type_count, *, zero_uniform=False):
    """Ordered values, not a mapping; no queue inspection or future information.

    Physical callers must supply a positive sum. The policy decoder may request
    the explicit all-zero -> uniform fallback for its bounded preference box.
    """
    if isinstance(values, Mapping):
        raise ValueError('task weights must be an ordered numeric vector')
    try:
        values = tuple(_number(v, 'task weight', True) for v in values)
    except TypeError as exc:
        raise ValueError('task weights must be an ordered numeric vector') from exc
    if len(values) != type_count or type_count < 1:
        raise ValueError('one task weight per configured type is required')
    largest = max(values)
    if largest == 0:
        if zero_uniform:
            return (1./type_count,)*type_count
        raise ValueError('physical task weights require a positive sum')
    # Keep normalized values stable through the decoder, environment and
    # scheduler. Scale only when the direct sum would overflow.
    total = sum(values)
    if math.isclose(total, 1., rel_tol=0., abs_tol=1e-12):
        return values
    if math.isfinite(total):
        return tuple(v/total for v in values)
    scaled = tuple(v/largest for v in values)
    total = math.fsum(scaled)
    return tuple(v/total for v in scaled)
