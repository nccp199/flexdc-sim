import pandas as pd

def calculate_qos_distribution(job_table):
    """
    Calculate QoS for started jobs using the selected logic.
    """
    started_jobs = job_table[
        (job_table['end_time'] > 0) &  # Ensure jobs have ended
        (job_table['arrival_time'] <= 18000)  # Filter jobs within the simulation duration
        ]
    if started_jobs.empty:
        return []

    # Calculate QoS for each job
    qos = (started_jobs['end_time'] - started_jobs['arrival_time'] - started_jobs['min_execution_time']) / started_jobs[
        'min_execution_time']
    job_table = job_table.assign(qos=qos)
    return job_table

def write_qos_summary_df(simulator, job_table=None, observed_until=None, suffix=''):
    """Write one fixed-cohort definition, including unresolved-task bounds."""
    import hashlib
    import json
    from pathlib import Path
    from peacsim.qos_evaluation import evaluate_qos, constraint_results

    if job_table is None:
        job_table = simulator.get_job_report()
    if observed_until is None:
        observed_until = simulator._simulation_duration
    constraints = simulator._job_config.all_job_qos_constraints
    minimum = simulator._job_config.all_min_execution_time
    start = getattr(simulator, '_qos_cohort_start', 0)
    end = getattr(simulator, '_qos_cohort_end', simulator._simulation_duration)
    tasks, summary = evaluate_qos(job_table, simulator._job_type_count, constraints, minimum,
                                  observed_until, start, end)
    cutoff = simulator.qos_violation_percentile_cutoff
    violations, ratio = constraint_results(summary, 1 - cutoff)
    summary['allowed_late_rate'] = 1 - cutoff
    summary['qos_violation'] = violations.to_numpy()

    # Retain the old columns for plotting, explicitly labeled as diagnostics.
    completed = tasks.loc[tasks.completed_as_of].copy()
    min_times = completed.job_type_id.map(minimum)
    qos = (completed.end_time - completed.arrival_time - min_times) / min_times
    p90 = qos.groupby(completed.job_type_id).quantile(cutoff)
    summary['qos_' + str(int(cutoff * 100)) + 'th'] = summary.job_type_id.map(p90)
    if 'start_time' in completed:
        slowdown = ((completed.end_time - completed.start_time - min_times) / min_times).groupby(
            completed.job_type_id).mean()
        summary['mean_exec_slowdown'] = summary.job_type_id.map(slowdown)
    else:
        summary['mean_exec_slowdown'] = float('nan')
    summary['percentile_scope'] = 'completed_cohort_tasks_only'
    output = Path(simulator._output_dir)
    summary.to_csv(output / ('qos_summary' + suffix + '.csv'), index=False)
    tasks.to_csv(output / ('qos_task_status' + suffix + '.csv'), index=False)
    identity_columns = [c for c in ['job_id', 'job_type_id', 'arrival_time',
                                    'min_execution_time', 'qos_constraint'] if c in tasks]
    identity = tasks[identity_columns].to_csv(index=False, float_format='%.17g', lineterminator='\n')
    metadata = dict(cohort_start=start, cohort_end=end, observed_until=observed_until,
                    total=int(summary.total.sum()), on_time=int(summary.on_time.sum()),
                    late=int(summary.late.sum()), pending=int(summary.pending.sum()),
                    final=bool((summary.evaluation_status == 'final').all()),
                    cohort_sha256=hashlib.sha256(identity.encode('utf-8')).hexdigest(),
                    deadline_rule='arrival_time + (1 + qos_constraint) * min_execution_time',
                    feasibility_metric='late / total; pending tasks have bounds but no final score',
                    legacy_diagnostics=['qos percentile columns', 'mean_exec_slowdown', 'QoSDegAvg.csv'])
    (output / ('qos_evaluation' + suffix + '.json')).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if not suffix:
        simulator.qos_summary = summary
    print('QoS cohort:', metadata)
    print(summary[['job_type_id', 'total', 'on_time', 'late', 'pending', 'late_rate']].to_string(index=False))
    return violations, ratio
