"""Pure S5a extraction: whitelist current fields, aggregate, then rescale.

The caller supplies the current external sample and already executed intervals,
not full external trajectories. Task IDs only join current tasks to their nodes;
IDs, RNG seeds, episode IDs and scenario hashes are never numerical features.
"""
import math

import numpy as np

from peacsim.energy.renewables import PV
from .specs import OBSERVATION_VERSION

EXTRACTION_VERSION = 'flexdc-observation-extraction-v2'

def task_details(native, job_config, endpoint, control_seconds):
    """Legal unfinished task rows shared by summaries and graph construction."""
    t = native['meta']['time_seconds']
    jobs, nodes = native['state']['jobs'], native['state']['nodes']
    # Recheck the allowed population even though the existing named view already
    # filters future tasks. The final native sample may start new endpoint jobs.
    arrived = ((jobs['arrival_time_seconds'] <= t)
               & (jobs['arrival_time_seconds'] < endpoint))
    unfinished = arrived & np.isin(jobs['status'], ['waiting', 'running'])
    indices = np.flatnonzero(unfinished)
    ids = jobs['job_id'][indices]
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate observed task IDs')
    # Accumulate over busy nodes once, so synchronous multi-node tasks have one
    # task count and node-weighted remaining work. Idle residuals are ignored.
    work_by_job, nodes_by_job = {}, {}
    for i in np.flatnonzero(nodes['is_busy']):
        # A released server can retain its previous task's progress until the
        # next physical cycle. A newly assigned task has made no progress yet.
        progress = 0. if nodes['just_started'][i] else float(nodes['task_progress_ratio'][i])
        minimum = float(nodes['task_min_execution_time_seconds'][i])
        if not math.isfinite(progress) or not math.isfinite(minimum) or minimum <= 0:
            raise ValueError('invalid current busy-node progress/profile')
        job_id = int(nodes['job_id'][i])
        work = minimum * (1. - min(1., max(0., progress)))
        work_by_job[job_id] = work_by_job.get(job_id, 0.) + work
        nodes_by_job[job_id] = nodes_by_job.get(job_id, 0) + 1
    rows, queue_counts = [], {}
    for i in indices:
        j, job_id = int(jobs['job_type_id'][i]), int(jobs['job_id'][i])
        size = job_config.all_job_size[j]
        minimum = float(jobs['min_execution_time_seconds'][i])
        waiting = jobs['status'][i] == 'waiting'
        if waiting:
            work = size * minimum
        else:
            if nodes_by_job.get(job_id) != size:
                raise ValueError('running task must retain its configured synchronous node group')
            work = work_by_job[job_id]
        slack = (float(jobs['arrival_time_seconds'][i])
                 + (1. + float(jobs['qos_allowance_ratio'][i])) * minimum - t)
        rank = queue_counts.get(j, 0) if waiting else 0
        if waiting:
            queue_counts[j] = rank + 1
        rows.append(dict(index=int(i), job_id=job_id, job_type=j, waiting=bool(waiting),
                         required_nodes=int(size), work=float(work), slack=float(slack),
                         urgent=bool(slack <= work/size + control_seconds), queue_rank=rank))
    return rows


def _task_statistics(native, job_config, endpoint, control_seconds):
    return [(r['job_type'], r['waiting'], r['work'], r['slack'], r['urgent'])
            for r in task_details(native, job_config, endpoint, control_seconds)]


def build_observations(*, native, job_config, environment, observation_end_seconds,
                       current_signals, battery_current, battery_limits,
                       last_energy_rows, spec):
    """Build detached arrays from a completed native boundary; never mutate inputs."""
    if spec['schema_version'] != OBSERVATION_VERSION:
        raise ValueError('unsupported observation schema')
    t = native['meta']['time_seconds']
    if t is None or not native['lifecycle']['started'] or not 0 <= t <= observation_end_seconds:
        raise ValueError('observation requires a valid initialized boundary')
    remaining = observation_end_seconds - t
    if (current_signals is None) != (remaining == 0):
        raise ValueError('current external signals must match the observation boundary')
    required_signals = {'grid_target_kw', 'pv_fraction', 'price_per_kwh', 'carbon_kg_per_kwh'}
    if remaining and not required_signals.issubset(current_signals):
        raise ValueError('missing current external signal; zero is only a terminal placeholder')
    if remaining and battery_limits is None:
        raise ValueError('unfinished episode requires current battery power limits')
    nodes, control = native['state']['nodes'], native['control']
    present = control['command_present']
    pending = float(np.sum((control['requested_power_w'][present]
                           - nodes['actual_power_w'][present]))) / 1000.
    facility = float(np.sum(nodes['actual_power_w'])) / 1000. * (1.+environment.cooling.power_ratio)
    signal = current_signals or {}
    previous_error, duration = 0., 0.
    for row in last_energy_rows:
        a, b = row['start_seconds'], row['end_seconds']
        if not 0 <= a < b <= t:
            raise ValueError('previous interval must be completed, not future information')
        previous_error += (b-a) * (row['power_kw']['grid_import'] - row['signals']['grid_target_kw'])
        duration += b-a
    if duration:
        if last_energy_rows[-1]['end_seconds'] != t:
            raise ValueError('previous interval is stale')
        previous_error /= duration
    raw = {
        'common.remaining_seconds': float(remaining),
        'common.grid_target_kw': float(signal.get('grid_target_kw', 0.)),
        'common.pv_available_kw': PV(environment.pv).available_power_kw(signal.get('pv_fraction', 0.)),
        'common.facility_load_kw': facility,
        'common.battery_soc': float(battery_current['soc']),
        'common.energy_to_initial_kwh': (environment.battery.capacity_kwh*environment.battery.initial_soc
                                       - battery_current['energy_kwh']),
        'common.charge_limit_kw': 0. if not remaining else battery_limits.charge_power_kw,
        'common.discharge_limit_kw': 0. if not remaining else battery_limits.discharge_power_kw,
        'common.pending_it_delta_kw': pending,
        'common.price_per_kwh': float(signal.get('price_per_kwh', 0.)),
        'common.carbon_kg_per_kwh': float(signal.get('carbon_kg_per_kwh', 0.)),
        'common.signals_valid': float(bool(remaining)),
        'common.previous_grid_error_kw': previous_error,
        'common.previous_interval_valid': float(bool(duration)),
        'compute.idle_nodes': float(np.count_nonzero(~nodes['is_busy'])),
    }
    rows = _task_statistics(native, job_config, observation_end_seconds, environment.control_interval_seconds)
    for j in spec['task_type_order']:
        group = [r for r in rows if r[0] == j]
        prefix = 'type_'+str(j)+'.'
        raw[prefix+'waiting_count'] = float(sum(r[1] for r in group))
        raw[prefix+'running_count'] = float(sum(not r[1] for r in group))
        raw[prefix+'waiting_work_node_seconds'] = math.fsum(r[2] for r in group if r[1])
        raw[prefix+'running_work_node_seconds'] = math.fsum(r[2] for r in group if not r[1])
        raw[prefix+'urgent_work_node_seconds'] = math.fsum(r[2] for r in group if r[4])
        raw[prefix+'min_deadline_slack_seconds'] = min((r[3] for r in group), default=0.)
        raw[prefix+'has_unfinished'] = float(bool(group))
    raw['tasks.remaining_work_node_seconds'] = math.fsum(r[2] for r in rows)
    raw['tasks.urgent_work_node_seconds'] = math.fsum(r[2] for r in rows if r[4])
    raw['tasks.min_deadline_slack_seconds'] = min((r[3] for r in rows), default=0.)
    raw['tasks.has_unfinished'] = float(bool(rows))
    if set(raw) != set(spec['fields']) or not all(math.isfinite(v) for v in raw.values()):
        raise ValueError('observation fields must match the schema and remain finite')

    def vector(role):
        values = np.asarray([raw[name]/spec['fields'][name]['scale']
                             for name in spec['layouts'][role]], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite normalized observation')
        return values

    return dict(schema_version=OBSERVATION_VERSION, extraction_version=EXTRACTION_VERSION,
                time_seconds=t, raw=raw,
                observations={role: vector(role) for role in spec['agent_order']},
                shared_state=vector('critic'))
