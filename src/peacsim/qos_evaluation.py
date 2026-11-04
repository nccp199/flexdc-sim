"""One fixed-cohort deadline definition for reports and delay-based objectives."""
import math

import numpy as np
import pandas as pd


def prepare_qos_evaluation(experiment, jobs):
    """Reserve the same observation horizon for every policy, before generating jobs.

    Existing configs evaluate arrivals in [0, original simulation_duration).
    Keep normal arrivals throughout observation. Round up to whole hours so the
    simulator's existing hourly energy/tracking accounting includes the tail.
    Saved configs retain both cohort bounds, so replay does not extend again.
    """
    start, end = experiment.qos_cohort_start, experiment.qos_cohort_end
    if not (0 <= start < end <= experiment.simulation_duration):
        raise ValueError('QoS cohort must satisfy 0 <= start < end <= simulation_duration')
    step = experiment.time_granularity
    if step <= 0:
        raise ValueError('time_granularity must be positive')
    allowances = [(1 + jobs.all_job_qos_constraints[j]) * jobs.all_min_execution_time[j]
                  for j in range(jobs.job_type_count)]
    if not allowances or not np.isfinite(allowances).all() or min(allowances) <= 0:
        raise ValueError('Job deadline allowances must be finite and positive')
    unit = math.lcm(3600, step)
    required = max(experiment.simulation_duration, end + max(allowances))
    experiment._simulation_duration = math.ceil(required / unit) * unit
    return experiment.simulation_duration


def evaluate_qos(job_table, job_type_count, qos_constraints, min_execution_times,
                 observed_until, cohort_start=0, cohort_end=None):
    """Return per-task states and per-type counts, without guessing pending outcomes.

    Completion exactly at the deadline succeeds. After processing the deadline
    timestamp, an unfinished task has missed it. Late completion stays late.
    A future completion in an archived table is hidden when evaluating as-of.
    """
    if cohort_end is None:
        cohort_end = observed_until
    if not np.isfinite([cohort_start, cohort_end, observed_until]).all():
        raise ValueError('Evaluation times must be finite')
    if not (0 <= cohort_start < cohort_end and observed_until >= 0):
        raise ValueError('Invalid cohort or observation interval')
    if job_type_count <= 0:
        raise ValueError('At least one job type is required')
    minimum = np.asarray([min_execution_times[j] for j in range(job_type_count)], dtype=float)
    limits = np.asarray([qos_constraints[j] for j in range(job_type_count)], dtype=float)
    if (not np.isfinite(minimum).all() or not np.isfinite(limits).all()
            or (minimum <= 0).any() or (limits < 0).any()):
        raise ValueError('Minimum durations must be positive and QoS limits nonnegative')
    jobs = job_table.loc[(job_table.arrival_time >= cohort_start)
                        & (job_table.arrival_time < cohort_end)].copy()
    arrivals = jobs.arrival_time.to_numpy(dtype=float)
    finishes = jobs.end_time.to_numpy(dtype=float)
    types = jobs.job_type_id.to_numpy(dtype=float)
    if (not np.isfinite(arrivals).all() or not np.isfinite(finishes).all()
            or not np.isfinite(types).all() or (types != np.floor(types)).any()
            or (types < 0).any() or (types >= job_type_count).any()
            or ((finishes < 0) & (finishes != -1)).any()
            or ((finishes >= 0) & (finishes < arrivals)).any()):
        raise ValueError('Invalid task timestamps or job type IDs in the evaluation cohort')
    type_ids = types.astype(int)
    deadlines = arrivals + (1 + limits[type_ids]) * minimum[type_ids]
    completed = (finishes >= 0) & (finishes <= observed_until)
    on_time = completed & (finishes <= deadlines)
    late = (completed & ~on_time) | (~completed & (observed_until >= deadlines))
    jobs['deadline'] = deadlines
    jobs['completed_as_of'] = completed
    jobs['qos_status'] = np.where(on_time, 'on_time', np.where(late, 'late', 'pending'))
    jobs['observed_until'] = observed_until
    rows = []
    for job_type in range(job_type_count):
        selected = jobs.loc[jobs.job_type_id == job_type, 'qos_status']
        total = len(selected)
        counts = {name: int((selected == name).sum()) for name in ['on_time', 'late', 'pending']}
        lower = counts['late'] / total if total else np.nan
        upper = (counts['late'] + counts['pending']) / total if total else np.nan
        rows.append(dict(job_type_id=job_type, total=total, **counts,
                         late_rate_lower=lower, late_rate_upper=upper,
                         late_rate=lower if total and counts['pending'] == 0 else np.nan,
                         evaluation_status=('no_samples' if not total else
                                            'pending' if counts['pending'] else 'final'),
                         qos_threshold=limits[job_type], cohort_start=cohort_start,
                         cohort_end=cohort_end, observed_until=observed_until))
    return jobs, pd.DataFrame(rows)


def final_late_rates(summary):
    """A scalar objective requires resolved, nonempty samples for every job type."""
    pending = summary.loc[summary.pending > 0, 'job_type_id'].tolist()
    empty = summary.loc[summary.total == 0, 'job_type_id'].tolist()
    if pending:
        raise ValueError(f'QoS evaluation still has pending tasks for types {pending}; extend observation')
    if empty:
        raise ValueError(f'QoS evaluation has no cohort samples for types {empty}; no final score is available')
    return summary.late_rate.tolist()


def constraint_results(summary, allowed_late_rate):
    """Unknown results cannot be reported as a feasible solution."""
    if not 0 <= allowed_late_rate <= 1:
        raise ValueError('Allowed late rate must be between zero and one')
    rates = summary.late_rate
    violations = pd.Series(pd.array([
        pd.NA if pd.isna(rate) else bool(rate > allowed_late_rate
            and not np.isclose(rate, allowed_late_rate, rtol=0, atol=1e-12))
        for rate in rates], dtype='boolean'), index=summary.job_type_id)
    ratio = float(violations.mean()) if violations.notna().all() else np.nan
    return violations, ratio
