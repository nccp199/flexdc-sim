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

def write_qos_summary_df(simulator):
    """
    Checking whether QoS is violated.
    Modified to include un-started and unfinished jobs.
    """
    violate_qos = []
    qos_constraints = simulator._job_config.all_job_qos_constraints
    # for job_type in range(self._job_type_count):
    #     qos_violate = 0
    #     current_job_type = self._job_table[
    #         (self._job_table['job_type_id'] == job_type) & (self._job_table['arrival_time'] <= time_length)]
    #     realtime_qos_list = list(current_job_type['realtime_qos'])
    #     if len(realtime_qos_list) > 0:
    #         qos_deg_sort = np.sort(realtime_qos_list)
    #         qos_deg_at_90 = qos_deg_sort[int(0.9 * len(qos_deg_sort))]
    #         if qos_deg_at_90 > qos_constraints[job_type]:
    #             qos_violate = 1
    #             print(f'Jobtype {job_type} Violate QoS. {qos_deg_at_90} > {qos_constraints[job_type]}')
    #     violate_qos.append(qos_violate)  # may be used for setting P,R of next phase.
    # qos_violations = sum(violate_qos)

    started_jobs = simulator._job_table[
        (simulator._job_table['end_time'] > simulator._idle_status) &
        (simulator._job_table['arrival_time'] < simulator._simulation_duration)]
    if started_jobs.empty:
        print('No started jobs')
        return
    qos = started_jobs.groupby('job_type_id').apply(
        lambda x: (x['end_time'] - x['arrival_time'] - x['min_execution_time']) / x['min_execution_time'])
    str_qos_viol_percentile_cutoff = str(int(simulator.qos_violation_percentile_cutoff * 100))
    print('QoS deg at ' + str_qos_viol_percentile_cutoff + ' by job type')
    qos_deg_at_percentile = qos.groupby('job_type_id').quantile(simulator.qos_violation_percentile_cutoff)

    constraint_violations = (qos_deg_at_percentile > pd.Series(list(qos_constraints.values())))
    job_type_qos_violation_count = constraint_violations.sum()
    job_type_qos_violation_ratio = job_type_qos_violation_count / simulator._job_type_count
    print(qos_deg_at_percentile.to_string())
    # print(f'==> num violations>5: {(q90deg > 5).sum()}')
    print('Mean exec slowdown by job type')
    mean_exec_slowdown = started_jobs.groupby('job_type_id').apply(
        lambda x: ((x['end_time'] - x['start_time'] - x['min_execution_time']) / x['min_execution_time']).mean())
    print(mean_exec_slowdown)
    qos_df = pd.DataFrame({
        'job_type_id': qos_deg_at_percentile.index,
        'qos_' + str_qos_viol_percentile_cutoff + 'th': qos_deg_at_percentile.values,
        'mean_exec_slowdown': mean_exec_slowdown.values,
        'qos_threshold': [qos_constraints[job_type] for job_type in qos_deg_at_percentile.index]
    })
    qos_df.to_csv(simulator._output_dir + 'qos_summary.csv', index=False, header=True)

    print(
        f'Count of job types with QoS violations from {simulator._job_type_count} job types: 'f'{job_type_qos_violation_count}')
    print(f'Ratio of job types with QoS violation: {job_type_qos_violation_ratio}')

    print('Violations:', ', '.join(['' for idx, violation in enumerate(violate_qos) if violation]))
    queued_jobs = simulator._job_table.loc[(simulator._job_table['arrival_time'] <= simulator._simulation_duration)]
    print(f'Ratio of all jobs violating QoS, queued in the first hour (from {len(queued_jobs)} jobs):',
          round((queued_jobs['realtime_qos'].values > queued_jobs['qos_constraint'].values).mean(), 7))


    return constraint_violations, job_type_qos_violation_ratio

