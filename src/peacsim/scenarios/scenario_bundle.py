"""S4a: prepare a reusable scenario without constructing a controller or simulator.

JSON paths are relative to the JSON containing them. Native workload generation
uses the existing global RNG temporarily and restores it in finally; preparation
must be sequential (or in separate processes), as for the existing Simulator.
The returned bundle contains full source inputs, NOT an allowed RL observation.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from peacsim.create_tables import generate_job_table, init_job_table, init_node_table
from peacsim.energy.config import (BatteryConfig, CoolingConfig, PVConfig,
                                  load_battery_config, load_cooling_config, load_pv_config)
from peacsim.energy.renewables import load_pv_profile
from peacsim.iso_signal import read_iso_signal
from peacsim.metrics.tracking import _number, _seconds
from peacsim.parsing.experiment_config_reader import ExperimentConfigReader
from peacsim.parsing.job_profile_reader import JobProfileReader
from .signals import Signal
from .dynamic_events import DynamicEvents


VERSION = 'flexdc-scenario-v1'


def _seed(value):
    value = _seconds(value, 'workload seed')
    if value >= 2**32:
        raise ValueError('workload seed must be below 2**32')
    return value


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _array_sha(array):
    return _sha(np.ascontiguousarray(array).tobytes())


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be explicit')
    return value


@dataclass(frozen=True)
class EnvironmentConfig:
    """Shared physical settings; no agent count, actions, reward or algorithm."""
    physical_step_seconds: int
    control_interval_seconds: int
    battery: BatteryConfig
    pv: PVConfig
    cooling: CoolingConfig
    grid_base_kw: float
    grid_reserve_kw: float
    grid_contract_source: str


class ScenarioBundle:
    """Read-only public interface. Mutable native arrays/configs are copy-out.

    Each future environment creates its own tasks, nodes, battery and ledger.
    No reset of a controller should reload files or redraw this bundle's inputs.
    """
    def __init__(self, *, scenario_id, end, cohort, environment, experiment,
                 jobs_config, job_table, node_table, signals, iso, metadata, dynamic=None):
        self._id, self._end, self._cohort = scenario_id, end, tuple(cohort)
        self._environment = environment
        self._experiment, self._jobs_config = deepcopy((experiment, jobs_config))
        # Immutable byte buffers prevent writable array views into shared inputs.
        self._jobs = (job_table.shape, np.asarray(job_table, dtype='<f8').tobytes())
        self._nodes = (node_table.shape, np.asarray(node_table, dtype='<f8').tobytes())
        self._signals, self._iso = dict(signals), tuple(iso)
        self._metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        self._dynamic = dynamic

    def operating_conditions(self, seconds):
        """Only effective settings at the requested boundary."""
        return None if self._dynamic is None else self._dynamic.at(seconds)

    @property
    def scenario_id(self):
        return self._id

    @property
    def observation_end_seconds(self):
        return self._end

    @property
    def cohort_seconds(self):
        return self._cohort

    @property
    def environment(self):
        return self._environment

    @property
    def initial_jobs(self):
        shape, data = self._jobs
        return np.frombuffer(data, dtype='<f8').reshape(shape).copy()

    @property
    def initial_nodes(self):
        shape, data = self._nodes
        return np.frombuffer(data, dtype='<f8').reshape(shape).copy()

    @property
    def experiment_config(self):
        return deepcopy(self._experiment)

    @property
    def job_config(self):
        return deepcopy(self._jobs_config)

    @property
    def iso_samples(self):
        """Includes the native terminal sample; no energy interval follows it."""
        return self._iso

    @property
    def metadata(self):
        return json.loads(self._metadata)

    def at(self, seconds):
        """External interval values for 0 <= t < T, not an agent observation."""
        seconds = _seconds(seconds, 'scenario time')
        if seconds >= self._end:
            raise ValueError('no energy interval starts at or after the observation endpoint')
        return {name: signal.at(seconds) for name, signal in self._signals.items()}

    def segments(self, start, end):
        """Split at the union of all signal boundaries, without averaging."""
        start, end = _seconds(start, 'segment start'), _seconds(end, 'segment end')
        if not 0 <= start < end <= self._end:
            raise ValueError('requested interval is outside the observation window')
        cuts = {start, end}
        for signal in self._signals.values():
            for left, right, _ in signal.segments(start, end):
                cuts.update((left, right))
        times = sorted(cuts)
        return tuple(dict(start_seconds=a, end_seconds=b, **self.at(a))
                     for a, b in zip(times, times[1:]))


def load_scenario(path, *, environment_config=None):
    """Prepare inputs only. An optional environment file reuses the same signals
    and workload recipe at another equipment scale; its own references resolve
    relative to itself. It is a different experimental condition, not an agent.
    """
    path = Path(path).resolve()
    files = {}

    def remember(p):
        p = p.resolve()
        files[str(p)] = _sha(p.read_bytes())
        return p

    def reference(owner, value):
        return remember(owner.parent / _text(value, 'input path'))

    def read_json(p):
        return json.loads(remember(p).read_text(encoding='utf-8'))

    spec = read_json(path)
    if spec['schema_version'] != VERSION:
        raise ValueError('unsupported scenario schema version')
    scenario_id = _text(spec['scenario_id'], 'scenario_id')
    _text(spec['source'], 'scenario source')
    env_path = (Path(environment_config).resolve() if environment_config is not None
                else reference(path, spec['environment_config']))
    env = read_json(env_path)
    if env['schema_version'] != 'flexdc-environment-config-v1':
        raise ValueError('unsupported environment config version')
    exp = ExperimentConfigReader(reference(env_path, env['native_experiment']))
    jobs = JobProfileReader(reference(path, spec['workload']['job_profile']))
    cohort = tuple(_seconds(v, 'cohort boundary') for v in spec['window']['cohort_seconds'])
    end = _seconds(spec['window']['observation_end_seconds'], 'observation end')
    if len(cohort) != 2 or not 0 <= cohort[0] < cohort[1] <= end:
        raise ValueError('cohort must lie inside the positive observation window')
    step = _seconds(exp.time_granularity, 'physical step')
    control = _seconds(env['control_interval_seconds'], 'control interval')
    if step == 0 or control == 0 or control % step or end % step:
        raise ValueError('positive control interval and full window must align to physical steps')
    if any(t % step for t in cohort):
        raise ValueError('cohort boundaries must align to physical steps')
    # Require the declared window rather than silently expanding it. This also
    # allows genuinely short hand-calculation fixtures without hourly rounding.
    allowances = [(1 + jobs.all_job_qos_constraints[j]) * jobs.all_min_execution_time[j]
                  for j in range(jobs.job_type_count)]
    if (not allowances or not np.isfinite(allowances).all() or min(allowances) <= 0
            or cohort[1] + max(allowances) > end):
        raise ValueError('observation tail must cover the cohort deadline allowance')
    if exp.server_count <= 0 or not np.isfinite(exp.idle_power) or exp.idle_power < 0:
        raise ValueError('invalid node count or idle power')
    for j in range(jobs.job_type_count):
        values = [jobs.all_min_execution_time[j], jobs.all_max_execution_time[j],
                  jobs.all_job_qos_constraints[j], jobs.all_min_job_power[j], jobs.all_max_job_power[j]]
        if (not np.isfinite(values).all() or not 1 <= jobs.all_job_size[j] <= exp.server_count
                or jobs.all_min_execution_time[j] <= 0
                or jobs.all_max_execution_time[j] < jobs.all_min_execution_time[j]
                or jobs.all_job_qos_constraints[j] < 0
                or not 0 <= jobs.all_min_job_power[j] <= jobs.all_max_job_power[j]):
            raise ValueError('invalid or unsupported job profile')
        if jobs.all_min_job_power[j] == jobs.all_max_job_power[j]:
            raise ValueError('job maximum power must exceed minimum power for the native linear progress model')
    workload = spec['workload']
    seed = _seed(workload['seed'])
    exp._random_seed, exp._simulation_duration = seed, end
    exp._qos_cohort_start, exp._qos_cohort_end = cohort
    iso_spec = spec['iso']
    iso_step = _seconds(iso_spec['interval_seconds'], 'ISO interval')
    if iso_step == 0:
        raise ValueError('ISO interval must be positive')
    iso_start = _seconds(iso_spec['start_hour'], 'ISO source start hour')
    if not isinstance(iso_spec['normalize'], bool):
        raise ValueError('ISO normalize must be boolean')
    exp._iso_file_path = str(reference(path, iso_spec['path']))
    exp._iso_signal_granularity = iso_step
    exp._iso_signal_start_hour = iso_start
    exp._randomize_iso_start = False
    exp._normalize_iso_signal = iso_spec['normalize']
    if iso_spec['normalize'] and end // iso_step == 0:
        raise ValueError('native normalization requires at least one full ISO interval')
    iso_source = _text(iso_spec['source'], 'ISO source')
    iso = tuple(float(v) for v in read_iso_signal(exp, end, iso_step, iso_spec['normalize']))
    if not np.isfinite(iso).all():
        raise ValueError('selected ISO samples must be finite')
    contract = env['grid_contract']
    base = _number(contract['base_kw'], 'grid base kW', True)
    reserve = _number(contract['reserve_kw'], 'grid reserve kW', True)
    contract_source = _text(contract['source'], 'grid contract source')
    targets = tuple(base + reserve * value for value in iso)
    if not np.isfinite(targets).all() or min(targets) < 0:
        raise ValueError('import-only grid targets must be finite and nonnegative')
    environment = EnvironmentConfig(
        step, control,
        load_battery_config(reference(env_path, env['battery_config'])),
        load_pv_config(reference(env_path, env['pv_config'])),
        load_cooling_config(reference(env_path, env['cooling_config'])),
        base, reserve, contract_source)
    pv_spec = spec['pv']
    pv = load_pv_profile(reference(path, pv_spec['profile']))
    offset = _seconds(pv_spec['source_offset_seconds'], 'PV source offset')
    pv.segments(offset, offset + end)  # Reuse the S3 coverage check.
    signals = {
        'iso': Signal(0, iso_step, iso, 'dimensionless', iso_source),
        'grid_target_kw': Signal(0, iso_step, targets, 'kW', contract_source),
        'pv_fraction': Signal(pv.start_seconds - offset, pv.interval_seconds,
                              pv.fractions, 'fraction', pv.source),
    }
    external = read_json(reference(path, spec['external_signals']))
    for name in ('price_per_kwh', 'carbon_kg_per_kwh'):
        signal = Signal(**external[name])
        if name == 'price_per_kwh' and (not signal.unit.endswith('/kWh') or not signal.unit[:-4].strip()):
            raise ValueError('price unit must state currency/kWh; no currency conversion')
        if name == 'carbon_kg_per_kwh' and signal.unit != 'kgCO2/kWh':
            raise ValueError('carbon intensity requires kgCO2/kWh')
        if name == 'carbon_kg_per_kwh' and min(signal.values) < 0:
            raise ValueError('average carbon intensity must be nonnegative')
        signal.segments(0, end)
        signals[name] = signal
    dynamic = None
    if 'dynamic_events' in spec:
        dynamic = DynamicEvents(read_json(reference(path, spec['dynamic_events'])),
                                step=step, end=end, job_types=jobs.job_type_count)
        price = signals['price_per_kwh']
        signals['price_per_kwh'] = Signal(
            0, step, tuple(price.at(t) * dynamic.price_multiplier(t) for t in range(0, end, step)),
            price.unit, price.source)
    # No stochastic preparation takes place until the external inputs validate.
    py_rng, np_rng = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        if workload['mode'] == 'native':
            if not np.isfinite(exp.utilization) or exp.utilization <= 0:
                raise ValueError('native workload utilization must be positive')
            table = init_job_table(exp, jobs)
        elif workload['mode'] == 'explicit':
            arrivals, types = [], []
            for row in workload['arrivals']:
                t = _seconds(row['time_seconds'], 'arrival time')
                j = _seconds(row['job_type'], 'job type')
                if t > end or j >= jobs.job_type_count or (arrivals and t < arrivals[-1]):
                    raise ValueError('explicit arrivals must be ordered, in-window and use valid job types')
                arrivals.append(t)
                types.append(j)
            table = generate_job_table(len(types), types, arrivals, jobs)
        else:
            raise ValueError('workload mode must be native or explicit')
        nodes = init_node_table(exp)
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
    if not np.isfinite(table).all():
        raise ValueError('initial task table must be finite')
    metadata = dict(
        schema_version=VERSION, scenario_id=scenario_id, source=spec['source'],
        observation_end_seconds=end, cohort_seconds=cohort, workload_seed=seed,
        scheduling_seed='supplied by the future run/controller configuration',
        environment=asdict(environment), server_count=exp.server_count,
        workload_mode=workload['mode'], task_count=table.shape[1],
        background_task_count=int(np.sum((table[2] < cohort[0]) | (table[2] >= cohort[1]))),
        initial_jobs_sha256=_array_sha(table), initial_nodes_sha256=_array_sha(nodes),
        selected_iso_sha256=_array_sha(np.asarray(iso, dtype='<f8')),
        iso_start_index=iso_start * 3600 // iso_step, iso_sample_count=len(iso),
        iso_normalization='native prefix maximum before slicing' if iso_spec['normalize'] else 'none',
        pv_source_offset_seconds=offset,
        signals={name: dict(unit=s.unit, source=s.source, interval_seconds=s.interval_seconds)
                 for name, s in signals.items()},
        input_sha256=files,
        endpoint='at(T) is invalid; iso_samples includes the native terminal sample',
    )
    # Stable across workspace relocation; distinguishes physical overrides even
    # when the human-readable scenario_id is unchanged.
    metadata['condition_sha256'] = _sha(json.dumps(dict(
        scenario=spec, environment=env, input_hashes=sorted(files.values()),
        initial_jobs_sha256=metadata['initial_jobs_sha256'],
        selected_iso_sha256=metadata['selected_iso_sha256']), sort_keys=True).encode())
    return ScenarioBundle(scenario_id=scenario_id, end=end, cohort=cohort,
                          environment=environment, experiment=exp, jobs_config=jobs,
                          job_table=table, node_table=nodes, signals=signals,
                          iso=iso, metadata=metadata, dynamic=dynamic)
