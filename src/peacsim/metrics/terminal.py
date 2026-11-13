"""S4e read-only terminal conditions; reuse the energy tolerance and QoS rules."""
import numpy as np

from peacsim.energy.config import DEFAULT_TOLERANCES
from peacsim.metrics.tracking import _number, _seconds
from peacsim.qos_evaluation import evaluate_qos


EPISODE_REPORT_VERSION = 'flexdc-episode-report-v1'
TERMINAL_ENERGY_RULE = 'final_energy_not_below_initial'


def terminal_energy_condition(initial_energy_kwh, final_energy_kwh):
    """Check the common inventory floor without moving energy or altering cost."""
    initial = _number(initial_energy_kwh, 'initial_energy_kwh', True)
    final = _number(final_energy_kwh, 'final_energy_kwh', True)
    tol = DEFAULT_TOLERANCES
    met = final >= initial or tol.cumulative_energy_matches(final, initial)
    return dict(rule=TERMINAL_ENERGY_RULE, initial_energy_kwh=initial, final_energy_kwh=final,
                net_change_kwh=final-initial, shortfall_kwh=max(0., initial-final),
                surplus_kwh=max(0., final-initial), condition_met=met,
                status='met' if met else 'not_met',
                tolerance=dict(absolute_kwh=tol.cumulative_energy_abs_kwh, relative=tol.relative),
                purchase_cost_adjustment=0.,
                interpretation='terminal battery condition only; assess task service and tracking separately')


def _backlog_stats(tasks):
    count = len(tasks)
    return dict(count=count, waiting_count=int((~tasks.started_as_of).sum()),
                running_count=int(tasks.started_as_of.sum()),
                queue_wait_sum_seconds=float(tasks.queue_wait_seconds.sum()),
                queue_wait_mean_seconds=float(tasks.queue_wait_seconds.mean()) if count else None,
                queue_wait_max_seconds=float(tasks.queue_wait_seconds.max()) if count else None,
                age_mean_seconds=float(tasks.age_seconds.mean()) if count else None,
                age_max_seconds=float(tasks.age_seconds.max()) if count else None)


def background_terminal_report(job_report, job_config, *, cohort_seconds, observed_until):
    """Classify arrived tasks outside the original cohort at the fixed endpoint.

    Population is [0, observed_until) minus the fixed evaluation cohort, including
    pre-cohort arrivals if any. Arrivals exactly at the final native sample have
    no elapsed service interval and are reported separately, not as backlog.
    Deadline equality and completed/late/pending statuses come from evaluate_qos.
    """
    start, end = (_seconds(v, 'cohort boundary') for v in cohort_seconds)
    observed = _seconds(observed_until, 'observed_until')
    if not 0 <= start < end <= observed:
        raise ValueError('terminal observation must cover the fixed cohort')
    outside = job_report.loc[(job_report.arrival_time < start) | (job_report.arrival_time >= end)]
    tasks, _ = evaluate_qos(outside, job_config.job_type_count, job_config.all_job_qos_constraints,
                            job_config.all_min_execution_time, observed, 0, observed)
    starts = tasks.start_time.to_numpy(dtype=float)
    arrivals = tasks.arrival_time.to_numpy(dtype=float)
    if (not np.isfinite(starts).all() or ((starts < 0) & (starts != -1)).any()
            or ((starts >= 0) & (starts < arrivals)).any()):
        raise ValueError('invalid background task start time')
    completed, late = tasks.completed_as_of, tasks.qos_status == 'late'
    tasks['terminal_status'] = np.select(
        [completed & ~late, completed & late, ~completed & late],
        ['completed_on_time', 'completed_late', 'overdue_unfinished'], default='not_due_unfinished')
    tasks['started_as_of'] = (starts >= 0) & (starts <= observed)
    tasks['queue_wait_seconds'] = np.where(tasks.started_as_of, starts, observed) - arrivals
    tasks['age_seconds'] = np.where(completed, tasks.end_time, observed) - arrivals
    statuses = ['completed_on_time', 'completed_late', 'overdue_unfinished', 'not_due_unfinished']
    counts = {name:int((tasks.terminal_status == name).sum()) for name in statuses}
    unfinished = tasks.loc[~completed]
    summary = dict(population='arrivals in [0, observation_end) outside the fixed evaluation cohort',
                   observation_end_seconds=observed, excluded_cohort_seconds=[start,end],
                   total=len(tasks), **counts,
                   deadline_violation_count=counts['completed_late']+counts['overdue_unfinished'],
                   unfinished_count=len(unfinished),
                   backlog=dict(all=_backlog_stats(unfinished),
                                overdue=_backlog_stats(tasks.loc[tasks.terminal_status == 'overdue_unfinished']),
                                not_due=_backlog_stats(tasks.loc[tasks.terminal_status == 'not_due_unfinished'])),
                   endpoint_arrival_count=int((job_report.arrival_time == observed).sum()),
                   future_arrival_count=int((job_report.arrival_time > observed).sum()),
                   not_due_backlog_policy='reported without penalty; reward weights deferred to S5',
                   deadline_rule='existing evaluate_qos; completed at deadline succeeds, unfinished at deadline is late')
    return tasks, summary
