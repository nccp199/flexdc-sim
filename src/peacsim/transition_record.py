"""One native cycle's observed phases; no dynamics, metrics, I/O or RNG.

Only the latest completed record is retained by Simulator. Public readers get
owned dictionaries, as with state snapshots. Full task tables stay in get_state.
This is diagnostic data, not an RL observation or an energy/tracking ledger.
"""
from copy import deepcopy

import numpy as np


SCHEMA_VERSION = 'flexdc-transition-v1'
# Original node-table rows. Keep the recorder independent of Simulator imports.
NODE_ID, JOB_ID, JOB_TYPE, POWER, _, PROGRESS, _, _, _, _, JUST_STARTED, COMMAND = range(12)


def _nodes(nodes):
    busy = nodes[JOB_TYPE] != -1
    just_started = nodes[JUST_STARTED] == 1
    progress_valid = busy & ~just_started
    present = nodes[COMMAND] != -1
    return dict(node_id=nodes[NODE_ID].astype(np.int64),
                job_id=nodes[JOB_ID].astype(np.int64),
                job_type_id=nodes[JOB_TYPE].astype(np.int64),
                is_busy=busy, just_started=just_started,
                actual_power_w=nodes[POWER].copy(),
                raw_node_progress_ratio=nodes[PROGRESS].copy(),
                task_progress_valid=progress_valid,
                task_progress_ratio=np.where(progress_valid, nodes[PROGRESS], np.nan),
                command_present=present,
                requested_power_w=np.where(present, nodes[COMMAND], np.nan))


class TransitionRecord:
    """Internal cycle-local collector; never retains the simulator or its arrays."""

    def __init__(self, sim, current_time, p_target):
        nodes = sim._node_table
        elapsed = current_time > 0
        duration = sim._time_granularity if elapsed else 0
        count = nodes.shape[1]
        self._data = dict(
            meta=dict(schema_version=SCHEMA_VERSION, episode_id=sim._episode_id,
                      sample_index=sim._next_cycle, time_seconds=current_time,
                      previous_sample_index=sim._next_cycle - 1 if elapsed else None,
                      kind='interval' if elapsed else 'initialization'),
            interval=dict(start_time_seconds=current_time - duration,
                          end_time_seconds=current_time, duration_seconds=duration,
                          has_elapsed_time=elapsed,
                          actual_power_w=nodes[POWER].copy() if elapsed else np.full(count, np.nan),
                          actual_it_power_w=float(np.sum(nodes[POWER])) if elapsed else None),
            before=_nodes(nodes),
            progress=dict(executed=False, updated_mask=np.zeros(count, dtype=bool),
                          skipped_idle_mask=np.zeros(count, dtype=bool),
                          skipped_just_started_mask=np.zeros(count, dtype=bool),
                          increment_ratio=np.zeros(count)),
            commands=dict(applied_mask=np.zeros(count, dtype=bool),
                          applied_power_w=np.full(count, np.nan),
                          actual_power_after_application_w=nodes[POWER].copy()),
            events=dict(completed_job_ids=np.empty(0, dtype=np.int64),
                        released_node_ids=np.empty(0, dtype=np.int64),
                        released_job_ids=np.empty(0, dtype=np.int64),
                        started_job_ids=np.empty(0, dtype=np.int64),
                        started_node_ids=np.empty(0, dtype=np.int64),
                        started_node_job_ids=np.empty(0, dtype=np.int64)),
            after_release=dict(actual_power_w=nodes[POWER].copy()),
            policy_input=dict(time_seconds=current_time, cluster_target_power_w=float(p_target)),
            after={},
        )

    def capture_progress(self, nodes, active_mask, just_started_mask, control_mask, increments):
        """Observe the masks and increments computed by the original core."""
        progress = self._data['progress']
        progress.update(executed=True, updated_mask=active_mask.copy(),
                        skipped_idle_mask=~self._data['before']['is_busy'],
                        skipped_just_started_mask=just_started_mask.copy())
        progress['increment_ratio'][active_mask] = increments
        self._data['commands'] = dict(
            applied_mask=control_mask.copy(),
            applied_power_w=np.where(control_mask, nodes[COMMAND], np.nan),
            actual_power_after_application_w=nodes[POWER].copy())

    def capture_completion(self, nodes, finished_servers, finished_jobs):
        """Copy the original completion decision before release overwrites IDs."""
        self._data['events'].update(
            completed_job_ids=finished_jobs.copy(),
            released_node_ids=finished_servers.copy(),
            released_job_ids=nodes[JOB_ID, finished_servers].astype(np.int64))

    def capture_after_release(self, nodes):
        self._data['after_release']['actual_power_w'] = nodes[POWER].copy()

    def finish(self, nodes):
        self._data['after'] = _nodes(nodes)
        started = nodes[JUST_STARTED] == 1
        self._data['events'].update(
            started_job_ids=np.unique(nodes[JOB_ID, started]).astype(np.int64),
            started_node_ids=nodes[NODE_ID, started].astype(np.int64),
            started_node_job_ids=nodes[JOB_ID, started].astype(np.int64))

    def as_dict(self):
        return deepcopy(self._data)

    def tracking_input(self):
        """Owned scalar projection for S2a; do not copy node arrays each step."""
        return dict(meta=self._data['meta'].copy(),
                    interval={key: value for key, value in self._data['interval'].items()
                              if key != 'actual_power_w'})


FIELDS = {}


def _field(path, description, dtype, unit='1', *, source, phase,
           valid_when='always', missing='none', axis=None):
    FIELDS[path] = dict(description=description, dtype=dtype, unit=unit,
                        source=source, phase=phase, valid_when=valid_when,
                        missing=missing, axis=axis)


for name, desc, dtype, unit, missing in [
    ('schema_version', '字段字典版本', 'str', '1', 'none'),
    ('episode_id', '与同一轮get_state的实验标识一致', 'str|None', '1', 'None=普通Simulator未由Session管理'),
    ('sample_index', '本次原生采样序号；与终点快照相同', 'int', '1', 'none'),
    ('time_seconds', '本轮成功处理的终点时刻；与终点快照相同', 'number', 's', 'none'),
    ('previous_sample_index', '本轮推进前的采样序号', 'int|None', '1', 'None=t=0初始化，无前一采样'),
    ('kind', 'initialization或interval；初始化不代表过去一秒', 'str', '1', 'none'),
]:
    _field('meta.' + name, desc, dtype, unit, source='Simulator clock/session',
           phase='successful cycle identity', missing=missing)

for name, desc in [
    ('start_time_seconds', '所记录推进段的起点；初始化为0'),
    ('end_time_seconds', '所记录推进段的终点；初始化为0'),
    ('duration_seconds', '实际推进的时长；初始化为0，之后沿用原生dt，不补尾部小步'),
]:
    _field('interval.' + name, desc, 'number', 's', source='native sample grid', phase='elapsed interval')
_field('interval.has_elapsed_time', '本次是否存在实际推进段', 'bool',
       source='current_time > 0', phase='elapsed interval')
for name, desc, dtype, missing, axis in [
    ('actual_power_w', '本段全部节点的实际功率；包含空闲和just_started节点，不按进度更新掩码过滤',
     'float64[]', 'NaN=t=0无推进段；0 W有效', 'nodes'),
    ('actual_it_power_w', '本段全部节点的实际功率和；不是终点新功率或设施总功率',
     'float|None', 'None=t=0无推进段；0 W有效', None),
]:
    _field('interval.' + name, desc, dtype, 'W', source='node server_power before progress/old command application',
           phase='constant over native elapsed interval', valid_when='interval.has_elapsed_time', missing=missing, axis=axis)

NODE_FIELDS = [
    ('node_id', '节点编号', 'int64[]', '1', 'node_id', 'always', 'none'),
    ('job_id', '节点正在执行的任务编号', 'int64[]', '1', 'server_job_id', 'is_busy', '-1=无任务'),
    ('job_type_id', '节点任务类型编号', 'int64[]', '1', 'server_state', 'is_busy', '-1=空闲'),
    ('is_busy', '节点是否忙碌，同时标记当前任务字段是否适用', 'bool[]', '1', 'server_state != -1', 'always', 'none'),
    ('just_started', '原刚启动标记；下一轮跳过进度，但仍有实际功率', 'bool[]', '1', 'just_started == 1', 'always', 'none'),
    ('actual_power_w', '该阶段的节点实际功率；含空闲节点', 'float64[]', 'W', 'server_power', 'always', 'none; 0 W有效'),
    ('raw_node_progress_ratio', '原节点表进度；空闲或刚重新分配时可能残留上一任务的值，不能直接当成当前任务进度', 'float64[]', '1', 'job_progress', 'always', 'none; raw values may exceed 1'),
    ('task_progress_valid', '节点忙碌且不是just_started时，原节点进度才适用于当前任务', 'bool[]', '1', 'server_state != -1 and just_started != 1', 'always', 'none'),
    ('task_progress_ratio', '当前任务有效的节点进度比例；保留原数值，不截断', 'float64[]', '1', 'job_progress masked by task_progress_valid', 'task_progress_valid', 'NaN=空闲或刚启动，不把残留进度当成新任务进度'),
    ('command_present', '该阶段是否存在待执行功率请求', 'bool[]', '1', 'control_power != -1', 'always', 'none'),
    ('requested_power_w', '该阶段待执行请求；不是区间实际功率', 'float64[]', 'W', 'control_power', 'command_present', 'NaN=无请求；0 W有效'),
]
for group, phase in [('before', 'initial table at t=0; otherwise previous post-scheduling boundary, before progress/commands'),
                     ('after', 'after scheduling at interval end')]:
    for name, desc, dtype, unit, source, valid, missing in NODE_FIELDS:
        _field(group + '.' + name, desc, dtype, unit, source='node_table/' + source, phase=phase,
               valid_when=valid if valid == 'always' else group + '.' + valid, missing=missing, axis='nodes')

_field('progress.executed', '本轮是否调用了进度更新；t=0不调用', 'bool',
       source='Simulator.update_job_progress hook', phase='before old command application')
for name, desc, source in [
    ('updated_mask', '实际进入原进度公式的节点', 'original active_mask'),
    ('skipped_idle_mask', '因空闲而不更新进度的节点', 'before.is_busy == False'),
    ('skipped_just_started_mask', '因刚启动而跳过进度的节点；与空闲掩码可重叠', 'original just_started_mask'),
]:
    _field('progress.' + name, desc, 'bool[]', source=source, phase='before old command application',
           valid_when='progress.executed', missing='all False=t=0未执行', axis='nodes')
_field('progress.increment_ratio', '原公式实际算出的增量；跳过节点为0，不把空闲残留值清零记作负进度',
       'float64[]', source='original active_job_progress, scattered by original active_mask',
       phase='before old command application', valid_when='progress.executed', missing='all 0=t=0未执行', axis='nodes')

_field('commands.applied_mask', '本轮实际进入旧指令赋值的节点；与before.command_present区别在t=0不应用',
       'bool[]', source='original control_power_mask', phase='after progress, before completion/release',
       missing='all False=t=0未应用', axis='nodes')
_field('commands.applied_power_w', '本轮应用的旧请求值；不保证任务释放或重新分配后仍为此功率',
       'float64[]', 'W', source='node_table/control_power at original assignment',
       phase='after progress, before completion/release', valid_when='commands.applied_mask',
       missing='NaN=未应用；0 W有效', axis='nodes')
_field('commands.actual_power_after_application_w', '旧指令应用后、完成释放前的全部节点实际功率；初始化保留初值',
       'float64[]', 'W', source='node_table/server_power after update_job_progress, initial power at t=0',
       phase='after old command application, before completion/release', axis='nodes')
_field('after_release.actual_power_w', '完成释放后、调度前的全部节点实际功率；新任务尚未启动',
       'float64[]', 'W', source='node_table/server_power after update_cluster',
       phase='after completion/release, before scheduling', axis='nodes')

for name, desc, source, phase, axis in [
    ('completed_job_ids', '原内核判定完成的任务ID，不另行修正多节点完成规则', 'check_all_other_servers_finished/finished_jobs', 'completion/release', 'completed_jobs'),
    ('released_node_ids', '原内核本轮释放的节点ID', 'check_all_other_servers_finished/finished_servers', 'completion/release', 'released_nodes'),
    ('released_job_ids', '释放前节点所归属的任务；与released_node_ids逐项对应，可重复', 'node_table/server_job_id before release', 'completion/release', 'released_nodes'),
    ('started_job_ids', '本轮新启动的任务ID去重集合', 'unique node_table/server_job_id where just_started == 1', 'after scheduling', 'started_jobs'),
    ('started_node_ids', '本轮分配给新任务的节点ID', 'node_table/node_id where just_started == 1', 'after scheduling', 'started_nodes'),
    ('started_node_job_ids', '新分配节点所归属的任务；与started_node_ids逐项对应，可重复', 'node_table/server_job_id where just_started == 1', 'after scheduling', 'started_nodes'),
]:
    _field('events.' + name, desc, 'int64[]', source=source, phase=phase,
           missing='empty array=no event', axis=axis)
_field('policy_input.time_seconds', '本轮策略调用时刻；不是区间目标的归属裁定', 'number', 's',
       source='Simulator current_time passed to execute', phase='policy input')
_field('policy_input.cluster_target_power_w', '本轮传给策略的目标功率；不是区间实际功率或已定义的正式跟踪目标',
       'float', 'W', source='resolved target series[current_time // iso_signal_granularity]', phase='policy input')


def get_transition_schema():
    return deepcopy(dict(
        version=SCHEMA_VERSION, fields=FIELDS,
        axes=dict(nodes='all node columns in original order, identified by before/after.node_id',
                  completed_jobs='core-selected completed job IDs', released_nodes='core-selected released nodes',
                  started_jobs='unique new job IDs', started_nodes='newly assigned nodes in original column order'),
        phase_order=['before', 'progress', 'commands', 'completion/release', 'after_release', 'policy_input', 'after'],
        ownership='owned arrays; no shared storage with engine, other fields, or previous reads',
        lifecycle='None before t=0/reset; latest successful cycle only; exhausted False/finalize create no record; failed episode rejects queries',
        scope='native AQA diagnostic phases; t=0 before commands may be raw initialization defaults and are not applied; not an RL observation, restorable checkpoint, or formal tracking/energy metric; borrowed trace durability belongs to its owner',
    ))
