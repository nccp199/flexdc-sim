"""Versioned, on-demand diagnostic snapshots. No stepping, I/O, RNG or caching.

Raw tables keep their exact layout/values. Named views identify roles and mask
inapplicable values. All arrays belong to the returned snapshot, not the engine.
The schema is separate so descriptions are not copied at every sampling time.
"""
from copy import deepcopy

import numpy as np

SCHEMA_VERSION = 'flexdc-state-v1'

# Each tuple: original name, unit, meaning, original validity/sentinel semantics.
JOB_SPECS = (
    ('job_id', '1', '任务编号', '始终有效；不是数组行号'),
    ('job_type_id', '1', '任务类型编号', '始终有效'),
    ('arrival_time', 's', '预生成的到达时刻', '未来任务也有值；不能作为已观测未来事实供智能体使用'),
    ('start_time', 's', '实际启动时刻', '-1=尚未启动；0 是有效时刻'),
    ('end_time', 's', '实际完成时刻', '-1=尚未完成；仿真结束不代表任务完成'),
    ('estimate_finished', '1', '任务表记录的完成进度比例，并非预计完成时刻',
     '初始化为0；运行中从节点更新，完成时置1；不代表每秒都会更新'),
    ('update_time', 's', '原任务记录的更新时间字段',
     '初始化为到达时间（可能在未来）；随后按原规则更新，不能据此证明未来任务已被观测'),
    ('realtime_qos', '1', '原过程QoS公式：(update_time-arrival_time)/min_execution_time-estimate_finished',
     '调度前计算；尚未处理t=0时仅为初值；并非最终QoS分类'),
    ('min_execution_time', 's', '任务类型的最短执行时间参数', '静态输入；不是剩余时间'),
    ('qos_constraint', '1', '允许相对最短执行时间增加的时间比例',
     '静态输入；对应截止时刻arrival_time+(1+qos_constraint)*min_execution_time'),
)
NODE_SPECS = (
    ('node_id', '1', '节点编号', '始终有效；不是数组行号'),
    ('server_job_id', '1', '正在执行的任务编号', '-1=无任务；结合server_state判断'),
    ('server_state', '1', '正在执行的任务类型编号', '-1=空闲；非-1为任务类型，非开关编码'),
    ('server_power', 'W', '当前节点实际功率', '含空闲功率；时点值，不是区间平均功率'),
    ('estimate_power', 'W', '原调度算法的功率估计工作值',
     '调度后采集；AQA中活跃节点通常等于请求功率；不是实测或独立预测'),
    ('job_progress', '1', '节点记录的任务进度比例',
     '仅忙碌节点适用于当前任务；空闲节点可能残留已完成任务的值，原值可大于1'),
    ('min_exec_time', 's', '节点所运行任务的最短执行时间参数', '仅忙碌节点适用；空闲可残留'),
    ('max_exec_time', 's', '节点所运行任务的最长执行时间参数', '仅忙碌节点适用；不是剩余时间'),
    ('min_job_power', 'W', '节点所运行任务的最低运行功率参数', '仅忙碌节点适用；不是节点空闲功率'),
    ('max_job_power', 'W', '节点所运行任务的最高运行功率参数', '仅忙碌节点适用；不是整机房上限'),
    ('just_started', '1', '原刚启动标志', '1=本轮新启动；-1=否；新任务有跳过首次进度更新的规则'),
    ('control_power', 'W', '待应用到节点的功率请求',
     '-1=无指令；0是数值指令；下一轮先推进进度再应用，末轮可能永不执行'),
)
JOB_FIELDS = tuple(s[0] for s in JOB_SPECS)
NODE_FIELDS = tuple(s[0] for s in NODE_SPECS)
JOB_PHASES = (
    'static input', 'static input', 'pregenerated scenario input',
    'set when assigned during scheduling', 'set on completion before scheduling',
    'updated from nodes/completion before scheduling; otherwise retained',
    'initially arrival_time; updated by cluster/QoS rules before scheduling',
    'computed before scheduling, not final QoS evaluation', 'static input', 'static input',
)
NODE_PHASES = (
    'static input', 'completion/release then scheduling', 'completion/release then scheduling',
    'previous command application, completion/release, or new task assignment; captured after scheduling',
    'policy working value after scheduling; initial value before first cycle',
    'progress update before command application; may be retained on release',
    'set on task assignment; retained when idle', 'set on task assignment; retained when idle',
    'set on task assignment; retained when idle', 'set on task assignment; retained when idle',
    'cleared before scheduling, set on new assignment',
    'cleared before scheduling, populated by policy; pending at capture',
)

FIELDS = {}


def _field(path, description, dtype, unit='1', *, source, phase='capture',
           valid_when='always', missing='none', axis=None, visibility='current_or_past'):
    FIELDS[path] = dict(description=description, dtype=dtype, unit=unit, source=source,
                        phase=phase, valid_when=valid_when, missing=missing,
                        axis=axis, visibility=visibility)


for name, desc, dtype, unit, missing in [
    ('schema_version', '字段字典版本', 'str', '1', 'none'),
    ('episode_id', '本次实验的独立输出目录标识；用于区分reset前后相同时间', 'str|None', '1', 'None=未交由Session管理'),
    ('time_seconds', '最近成功完成的采样时刻；不是所有字段的更新时间', 'number|None', 's', 'None=尚未处理t=0'),
    ('sample_index', '最近成功完成的采样序号，从0开始', 'int|None', '1', 'None=尚未处理t=0'),
    ('dt_seconds', '底层物理步长', 'number', 's', 'none'),
    ('next_sample_seconds', '下一轮将处理的采样时刻', 'number|None', 's', 'None=已无下一轮'),
    ('phase', 'initial或post_scheduling；只允许在成功轮次之间采集', 'str', '1', 'none'),
]:
    _field('meta.' + name, desc, dtype, unit, source='Simulator clock/session', missing=missing)
for name, desc in [
    ('started', '是否成功处理过t=0'), ('can_advance', '是否仍有采样轮次'),
    ('simulation_finished', '是否走完规定采样窗口；不是全部任务完成'),
    ('finalized', '最终报告是否成功结算并发布'),
]:
    _field('lifecycle.' + name, desc, 'bool', source='Simulator lifecycle')

JOB_VIEW = {
    'job_id': (0, 'int64'), 'job_type_id': (1, 'int64'),
    'arrival_time_seconds': (2, 'float64'), 'start_time_seconds': (3, 'float64'),
    'end_time_seconds': (4, 'float64'), 'recorded_progress_ratio': (5, 'float64'),
    'min_execution_time_seconds': (8, 'float64'), 'qos_allowance_ratio': (9, 'float64'),
}
for name, (row, dtype) in JOB_VIEW.items():
    original, unit, desc, validity = JOB_SPECS[row]
    named_validity = 'arrived by meta.time_seconds'
    if row == 3:
        named_validity += '; state.jobs.status != waiting'
    elif row == 4:
        named_validity += '; state.jobs.status == completed'
    _field('state.jobs.' + name, desc, dtype + '[]', unit, source='raw.job_table/' + original,
           phase=JOB_PHASES[row],
           valid_when=named_validity,
           missing={3: 'NaN=not started', 4: 'NaN=not completed'}.get(row, 'none'), axis='arrived_jobs')
_field('state.jobs.status', 'waiting/running/completed；依据启动和完成时刻分类', 'str[]',
       source='copied task arrival/start/end fields', axis='arrived_jobs')

NODE_VIEW = {
    'node_id': (0, 'int64'), 'job_id': (1, 'int64'), 'job_type_id': (2, 'int64'),
    'actual_power_w': (3, 'float64'), 'task_progress_ratio': (5, 'float64'),
    'task_min_execution_time_seconds': (6, 'float64'),
    'task_max_execution_time_seconds': (7, 'float64'),
    'task_min_power_w': (8, 'float64'), 'task_max_power_w': (9, 'float64'),
}
for name, (row, dtype) in NODE_VIEW.items():
    original, unit, desc, validity = NODE_SPECS[row]
    _field('state.nodes.' + name, desc, dtype + '[]', unit, source='raw.node_table/' + original,
           phase=NODE_PHASES[row], valid_when=validity, missing='NaN=idle/inapplicable' if row >= 5 else
           ('-1=idle/no task' if row in (1, 2) else 'none'), axis='nodes')
for name, desc in [('is_busy', '是否执行任务，同时是task_*字段的有效标记'),
                   ('just_started', '是否本轮刚启动；保留原跳过进度更新语义')]:
    _field('state.nodes.' + name, desc, 'bool[]', source='raw.node_table/server_state,just_started', axis='nodes')

_field('control.cluster_target_power_w', '最近一轮调度使用的集群功率目标；不是实际功率',
       'float|None', 'W', source='last successful policy input (external budget or resolved ISO target)', phase='policy input',
       valid_when='lifecycle.started', missing='None=not started')
_field('control.command_present', '是否存在功率指令；与是否还会推进下一轮分开判断',
       'bool[]', source='raw.node_table/control_power != -1', axis='nodes')
_field('control.requested_power_w', '待生效请求；不会反向修改过去区间', 'float64[]', 'W',
       source='raw.node_table/control_power', phase='pending at capture',
       valid_when='control.command_present', missing='NaN=no command', axis='nodes')
_field('control.application_phase', '生效顺序说明；不虚构实际应用时间', 'str',
       source='Simulator.update_job_progress order', phase='next cycle if one exists')

for name in ('waiting_jobs', 'running_jobs', 'completed_jobs'):
    _field('derived.' + name, '调度后已到达任务的' + name + '数量', 'int|None',
           source='count state.jobs.status', phase='capture',
           valid_when='lifecycle.started', missing='None=not started; no arrived view yet')
for name in ('busy_nodes', 'idle_nodes'):
    _field('derived.' + name, '快照节点数量：' + name, 'int', source='count state.nodes.is_busy')
for name, desc in [('actual_it_power_w', '所有节点的时点实际功率和，包含空闲节点'),
                   ('busy_node_power_w', '忙碌节点的时点实际功率和'),
                   ('idle_node_power_w', '空闲节点的时点实际功率和')]:
    _field('derived.' + name, desc, 'float', 'W', source='sum copied node server_power')

_field('diagnostics.policy_estimated_power_w', NODE_SPECS[4][2], 'float64[]', 'W',
       source='raw.node_table/estimate_power', phase='policy working value at capture',
       valid_when='lifecycle.started', missing='NaN=not started', axis='nodes')
_field('diagnostics.task_record_update_time_seconds', JOB_SPECS[6][2], 'float64[]', 's',
       source='raw.job_table/update_time, arrived rows only', phase='per-task recorded time', axis='arrived_jobs')
_field('diagnostics.legacy_realtime_qos', JOB_SPECS[7][2], 'float64[]',
       source='raw.job_table/realtime_qos, arrived rows only', phase='before scheduling', axis='arrived_jobs')
LEGACY_SCALARS = {
    'waiting_jobs_before_scheduling': ('waiting_sum', 'count', '本轮调度前的等待数量', 'before scheduling'),
    'qos_emergency_before_scheduling': ('qos_emergency', '1', '原调度前紧急标志，不是最终违约判定', 'before scheduling'),
    'tracking_error_mixed_time': ('tracking_error', '1', '原跟踪指标混合时间口径，保留到S2修正', 'previous power and lagged target'),
    'tracking_node_delta': ('tracking_num', 'count', '原目标差额折算节点数，可为负；不是实际队列长度', 'after scheduling'),
    'legacy_standby_power_w': ('standby_power', 'W', '原诊断值目前固定为0；不是空闲节点实际功率和', 'before scheduling'),
    'legacy_qos_job_sum_w': ('qos_job_sum', 'W', '原值等于节点实际总功率；不是QoS任务子集的功率', 'after scheduling'),
}
for name, (attr, unit, desc, phase) in LEGACY_SCALARS.items():
    _field('diagnostics.' + name, desc, 'number|bool|None', unit, source='Simulator.' + attr,
           phase=phase,
           valid_when=('lifecycle.started and nonzero legacy W/node conversion' if name == 'tracking_node_delta'
                       else 'lifecycle.started'),
           missing=('None=not started or undefined zero W/node conversion' if name == 'tracking_node_delta'
                    else 'None=not started'))
for kind, names in [('job', JOB_FIELDS), ('node', NODE_FIELDS)]:
    _field('raw.' + kind + '_table', '未经重命名、清洗或取整的完整原始数组副本', 'float64[fields, entities]',
           source='Simulator._' + kind + '_table', phase='capture; individual fields retain original update phases',
           axis='field rows, entity columns', visibility='diagnostic_only_includes_future' if kind == 'job' else 'current_or_past')
    _field('raw.' + kind + '_fields', '原始数组行名，顺序固定；列轴由ID行标识', 'tuple[str]', source='fixed raw schema')


def get_state_schema():
    raw = {}
    for kind, specs, phases in [('job', JOB_SPECS, JOB_PHASES), ('node', NODE_SPECS, NODE_PHASES)]:
        raw[kind] = [dict(row=i, name=name, unit=unit, dtype='float64',
                          description=desc, validity=validity, source='Simulator._' + kind + '_table[' + str(i) + ']',
                          phase=phases[i],
                          visibility='diagnostic_only_includes_future' if kind == 'job' else 'current_or_past')
                     for i, (name, unit, desc, validity) in enumerate(specs)]
    return deepcopy(dict(version=SCHEMA_VERSION, fields=FIELDS, raw_fields=raw,
                         axes=dict(nodes='all raw node columns, in original order',
                                   arrived_jobs='raw job columns with arrival_time<=time_seconds, in original order; empty before t=0'),
                         ownership='all arrays detached from engine and previous snapshots; named/raw arrays also independent',
                         scope='diagnostic snapshot, not RL observation or restorable checkpoint'))


def snapshot(sim):
    jobs, nodes = sim._job_table.copy(), sim._node_table.copy()
    started = sim._next_cycle > 0
    t = (sim._next_cycle - 1) * sim._time_granularity if started else None
    can_advance = sim._next_cycle <= sim._simulation_duration // sim._time_granularity
    arrived = jobs[:, jobs[2] <= t] if started else jobs[:, :0].copy()
    status = np.where(arrived[4] != -1, 'completed', np.where(arrived[3] != -1, 'running', 'waiting'))
    task_view = {name: arrived[row].astype(dtype, copy=True) for name, (row, dtype) in JOB_VIEW.items()}
    for key in ('start_time_seconds', 'end_time_seconds'):
        task_view[key][task_view[key] == -1] = np.nan
    task_view['status'] = status
    busy = nodes[2] != -1
    node_view = {name: nodes[row].astype(dtype, copy=True) for name, (row, dtype) in NODE_VIEW.items()}
    for name, (row, _) in NODE_VIEW.items():
        if row >= 5:
            node_view[name][~busy] = np.nan
    node_view.update(is_busy=busy, just_started=nodes[10] == 1)
    commands = nodes[11] != -1
    diagnostics = dict(policy_estimated_power_w=nodes[4].copy() if started else np.full(nodes.shape[1], np.nan),
                       task_record_update_time_seconds=arrived[6].copy(),
                       legacy_realtime_qos=arrived[7].copy())
    diagnostics.update({name: getattr(sim, spec[0]) if started else None
                        for name, spec in LEGACY_SCALARS.items()})
    derived = {name + '_jobs': int(np.count_nonzero(status == name)) if started else None
               for name in ('waiting', 'running', 'completed')}
    derived.update(busy_nodes=int(np.count_nonzero(busy)), idle_nodes=int(np.count_nonzero(~busy)),
                   actual_it_power_w=float(nodes[3].sum()),
                   busy_node_power_w=float(nodes[3, busy].sum()), idle_node_power_w=float(nodes[3, ~busy].sum()))
    return dict(
        meta=dict(schema_version=SCHEMA_VERSION, episode_id=sim._episode_id, time_seconds=t,
                  sample_index=sim._next_cycle - 1 if started else None, dt_seconds=sim._time_granularity,
                  next_sample_seconds=sim._next_cycle * sim._time_granularity if can_advance else None,
                  phase='post_scheduling' if started else 'initial'),
        lifecycle=dict(started=started, can_advance=can_advance, simulation_finished=not can_advance,
                       finalized=sim._run_finalized),
        state=dict(jobs=task_view, nodes=node_view),
        control=dict(cluster_target_power_w=sim._last_policy_target_w
                     if started else None, command_present=commands,
                     requested_power_w=np.where(commands, nodes[11], np.nan),
                     application_phase='next cycle: after progress update, before completion/release; no guarantee at final sample'),
        derived=derived, diagnostics=diagnostics,
        raw=dict(job_table=jobs, node_table=nodes, job_fields=JOB_FIELDS, node_fields=NODE_FIELDS))
