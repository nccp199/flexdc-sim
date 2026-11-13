"""S5a observation contract. Scales use declared configuration, never traces.

This module specifies information only. It does not define actions, rewards,
Gym spaces, a training algorithm or a graph with privileged extra information.
"""
import math


OBSERVATION_VERSION = 'flexdc-observation-v1'
AGENTS = ('compute', 'storage')


def observation_spec(*, job_config, server_count, idle_power_w, environment,
                     observation_end_seconds, price_unit):
    """Return an owned, JSON-serializable field dictionary and fixed layouts."""
    count = int(server_count)
    types = tuple(range(job_config.job_type_count))
    if count <= 0 or not types or observation_end_seconds <= 0:
        raise ValueError('observations require nodes, task types and a positive window')
    time_scale = max(float(job_config.all_min_execution_time[j]) for j in types)
    power_scale = max(1., count * max(float(idle_power_w),
                                     *job_config.all_max_job_power.values()) / 1000.)
    scales = dict(nodes=float(count), time_seconds=time_scale,
                  work_node_seconds=count*time_scale, power_kw=power_scale,
                  energy_kwh=environment.battery.capacity_kwh,
                  episode_seconds=float(observation_end_seconds))
    if not all(math.isfinite(v) and v > 0 for v in scales.values()):
        raise ValueError('observation scales must be positive and finite')
    fields = {}
    common, compute, storage = [], [], []

    def add(layout, name, unit, scale, description, source, *, raw_range='finite',
            phase='current decision boundary', valid_when='always', missing='none'):
        fields[name] = dict(unit=unit, scale=float(scale), description=description,
                            source=source, raw_range=raw_range, phase=phase,
                            valid_when=valid_when, missing=missing,
                            normalization='raw / scale; no fitting or clipping')
        layout.append(name)

    def c(name, unit, scale, description, source, **kwargs):
        add(common, 'common.'+name, unit, scale, description, source, **kwargs)

    c('remaining_seconds', 's', scales['episode_seconds'], '观察窗口剩余时间',
      'declared observation endpoint - native.meta.time_seconds', raw_range='[0, endpoint]')
    signal_rule = dict(valid_when='common.signals_valid == 1',
                       missing='0 at endpoint; signals_valid=0')
    c('grid_target_kw', 'kW', power_scale, '当前并网目标', 'scenario.at(current time).grid_target_kw',
      raw_range='>=0', **signal_rule)
    c('pv_available_kw', 'kW', power_scale, '当前可用光伏；不是未来实际利用量',
      'PV.available_power_kw(current pv_fraction)', raw_range='[0, declared PV peak]', **signal_rule)
    c('facility_load_kw', 'kW', power_scale, '当前IT时点功率加固定比例冷却；不是下一段真实负荷',
      'sum(native.state.nodes.actual_power_w) / 1000 * (1 + cooling.power_ratio)', raw_range='>=0')
    c('battery_soc', '1', 1., '当前电池SOC', 'FacilityEnergy.summary().battery_current.soc',
      raw_range='[configured soc_min, soc_max]')
    c('energy_to_initial_kwh', 'kWh', scales['energy_kwh'], '初始电量减当前电量；负值表示盈余',
      'battery.capacity_kwh * battery.initial_soc - battery_current.energy_kwh')
    for direction in ('charge', 'discharge'):
        c(direction+'_limit_kw', 'kW', power_scale, '本次保持时长下的电池'+direction+'功率上限',
          'existing Battery.power_limits(min(control interval, remaining time))', raw_range='>=0',
          missing='0 at endpoint', valid_when='remaining_seconds > 0; otherwise 0')
    c('pending_it_delta_kw', 'kW', power_scale, '待生效节点命令相对当前功率的合计差额',
      'sum((requested_power_w - actual_power_w)[command_present]) / 1000',
      phase='pending at current boundary; no promise of actual next load')
    c('price_per_kwh', price_unit, 1., '当前购电单价；保留负电价',
      'scenario.at(current time).price_per_kwh', **signal_rule)
    c('carbon_kg_per_kwh', 'kgCO2/kWh', 1., '当前外部平均碳强度；不是边际排放因子',
      'scenario.at(current time).carbon_kg_per_kwh', raw_range='>=0', **signal_rule)
    c('signals_valid', '1', 1., '当前时刻是否还有合法外部信号区间',
      'current time < declared endpoint', raw_range='{0,1}')
    c('previous_grid_error_kw', 'kW', power_scale, '上次推进区间的时间加权有符号并网误差',
      'last_energy_rows: sum((grid_import - matching grid_target_kw) * seconds) / seconds',
      phase='previous completed step/control interval', valid_when='common.previous_interval_valid == 1',
      missing='0 before any physical interval; previous_interval_valid=0')
    c('previous_interval_valid', '1', 1., '是否已经完成过物理区间',
      'nonempty last_energy_rows', raw_range='{0,1}')

    add(compute, 'compute.idle_nodes', 'node', count, '当前空闲节点数',
        'count(~native.state.nodes.is_busy)', raw_range='[0, server_count]')
    task_source = ('arrived unfinished tasks + current busy node progress + known task profile; '
                   'no raw future task table or recorded_progress_ratio')
    for j in types:
        prefix = 'type_'+str(j)+'.'
        for status in ('waiting', 'running'):
            add(compute, prefix+status+'_count', 'task', count, status+'任务数（每任务一次）',
                task_source, raw_range='>=0; waiting count may exceed node count')
        for name, description in [('waiting_work_node_seconds', '排队任务基准工作量'),
                                  ('running_work_node_seconds', '运行任务剩余基准工作量'),
                                  ('urgent_work_node_seconds', '紧急任务剩余基准工作量')]:
            add(compute, prefix+name, 'node*s', scales['work_node_seconds'], description,
                task_source, raw_range='>=0; not capped at scale')
        add(compute, prefix+'min_deadline_slack_seconds', 's', time_scale,
            '该类型未完成任务的最小截止余量；负值表示已过截止',
            'min(arrival + (1+qos_allowance) * min_execution_time - current time)',
            valid_when=prefix+'has_unfinished == 1', missing='0 when no unfinished task; mask=0')
        add(compute, prefix+'has_unfinished', '1', 1., '该类型是否有未完成任务',
            task_source, raw_range='{0,1}')
    for name, description in [('remaining_work_node_seconds', '所有未完成任务的剩余基准工作量'),
                              ('urgent_work_node_seconds', '所有紧急任务的剩余基准工作量')]:
        add(storage, 'tasks.'+name, 'node*s', scales['work_node_seconds'], description,
            task_source, raw_range='>=0; not capped at scale')
    add(storage, 'tasks.min_deadline_slack_seconds', 's', time_scale, '所有未完成任务的最小截止余量',
        'min of current unfinished task deadline slack', valid_when='tasks.has_unfinished == 1',
        missing='0 when no unfinished task; mask=0')
    add(storage, 'tasks.has_unfinished', '1', 1., '任务摘要有效标记', task_source, raw_range='{0,1}')

    layouts = dict(compute=common+compute, storage=common+storage,
                   critic=common+compute+storage)
    for name, field in fields.items():
        field['visible_to'] = [role for role, names in layouts.items() if name in names]
    return dict(schema_version=OBSERVATION_VERSION, agent_order=list(AGENTS),
                task_type_order=list(types), layouts=layouts, fields=fields, scales=scales,
                dimensions={role: len(names) for role, names in layouts.items()},
                dtype='float64', normalization_fitted=False,
                urgency_rule='deadline slack <= remaining baseline work / task node count + control interval',
                task_population='arrived and unfinished; arrivals exactly at final endpoint excluded',
                information_scope='fixed summaries, not a claim of a complete Markov state; no task-level graph privileges')
