"""S5c diagnostic-only team reward; consumes existing executed rows/reports.

No simulation mutation, future task lookup, automatic default config or trainer.
One append is one reward-bearing transition. The final append MUST include the
existing complete episode report, so terminal terms enter that transition once.
The sum is an undiscounted finite-window cost, not the legacy SA softplus score.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path

from peacsim.metrics.tracking import _number, _seconds, _run_id
from peacsim.metrics.terminal import EPISODE_REPORT_VERSION, terminal_energy_condition
from peacsim.rl.specs import AGENTS, observation_spec


REWARD_VERSION = 'flexdc-team-reward-v1'
CONFIG_VERSION = 'flexdc-reward-config-v1'
FROZEN_CONFIG_VERSION = 'flexdc-reward-config-v2'
PERIODIC = ('grid_tracking', 'grid_cost', 'grid_carbon')
TERMINAL = ('cohort_qos', 'background_qos', 'terminal_energy', 'not_due_backlog')


def _positive(value, name):
    value = _number(value, name, True)
    if value == 0:
        raise ValueError(name+' must be positive')
    return value


@dataclass(frozen=True)
class RewardWeights:
    grid_tracking: float
    cohort_qos: float
    background_qos: float
    terminal_energy: float
    grid_cost: float
    grid_carbon: float
    not_due_backlog: float

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self,field.name,_number(getattr(self,field.name),field.name,True))


@dataclass(frozen=True)
class RewardConfig:
    weights: RewardWeights
    reference_price_per_kwh: float
    reference_carbon_kg_per_kwh: float
    notes: str
    schema_version: str = CONFIG_VERSION
    purpose: str = 'diagnostic_only'
    final_weights: bool = False
    discount_factor: float = 1.

    def __post_init__(self):
        if not isinstance(self.weights,RewardWeights):
            raise TypeError('weights must be RewardWeights')
        diagnostic=(self.schema_version==CONFIG_VERSION and self.purpose=='diagnostic_only' and self.final_weights is False)
        frozen=(self.schema_version==FROZEN_CONFIG_VERSION and self.purpose=='frozen_experiment_protocol' and self.final_weights is True)
        if not (diagnostic or frozen):
            raise ValueError('reward requires explicit diagnostic v1 or frozen-protocol v2 identity')
        if _number(self.discount_factor,'discount_factor') != 1.:
            raise ValueError('this diagnostic objective is finite-window and undiscounted (gamma=1)')
        for name in ('reference_price_per_kwh','reference_carbon_kg_per_kwh'):
            object.__setattr__(self,name,_positive(getattr(self,name),name))
        if not isinstance(self.notes,str) or not self.notes.strip():
            raise ValueError('explicit test-only configuration notes are required')


def load_reward_config(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    data['weights'] = RewardWeights(**data['weights'])
    return RewardConfig(**data)


def reward_spec(scenario, config):
    """Declared static scales only; reuse S5a's node/power/energy references."""
    if not isinstance(config,RewardConfig):
        raise TypeError('an explicit RewardConfig is required')
    exp = scenario.experiment_config
    price_unit = scenario.metadata['signals']['price_per_kwh']['unit']
    scales = observation_spec(job_config=scenario.job_config,server_count=exp.server_count,
        idle_power_w=exp.idle_power,environment=scenario.environment,
        observation_end_seconds=scenario.observation_end_seconds,price_unit=price_unit)['scales']
    energy_ref = scales['power_kw']*scales['episode_seconds']/3600.
    denominators = dict(grid_tracking=scales['power_kw']*scales['episode_seconds'],
        grid_cost=energy_ref*config.reference_price_per_kwh,
        grid_carbon=energy_ref*config.reference_carbon_kg_per_kwh,
        terminal_energy=scales['energy_kwh'],not_due_backlog=scales['nodes'])
    for name,value in denominators.items():
        _positive(value,name+' normalizer')
    sources = dict(grid_tracking='sum(abs(actual grid_import - matching grid_target_kw) * seconds)',
        grid_cost='sum(existing grid_purchase_cost.amount)',grid_carbon='sum(existing grid_carbon_kg)',
        cohort_qos='existing terminal cohort_qos.late / total; task-weighted aggregate',
        background_qos='existing background.deadline_violation_count / total',
        terminal_energy='existing terminal shortfall_kwh if condition_met is false, otherwise zero',
        not_due_backlog='existing background.not_due_unfinished; not a deadline violation')
    units = dict(grid_tracking='kW*s',grid_cost=price_unit[:-4],grid_carbon='kgCO2',
                 cohort_qos='task',background_qos='task',terminal_energy='kWh',not_due_backlog='task')
    cfg = asdict(config)
    digest = hashlib.sha256(json.dumps(cfg,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return dict(schema_version=REWARD_VERSION,purpose=config.purpose,final_weights=config.final_weights,
        approved_for_formal_experiments=config.final_weights,config_sha256=digest,config=cfg,agent_order=list(AGENTS),
        start_seconds=0,end_seconds=scenario.observation_end_seconds,cohort_seconds=list(scenario.cohort_seconds),
        currency=price_unit[:-4],initial_energy_kwh=scenario.environment.battery.capacity_kwh*
            scenario.environment.battery.initial_soc,reference_scales=scales,
        components={name:dict(weight=weight,phase='periodic' if name in PERIODIC else 'terminal',
            raw_unit=units[name],normalizer=denominators.get(name),source=sources[name],
            normalizer_rule=('terminal population count; empty population contributes zero with has_samples=false'
                             if name in ('cohort_qos','background_qos') else 'declared static scale'))
            for name,weight in asdict(config.weights).items()},
        objective='negative sum of weighted normalized costs over the full window',
        discount_factor=1.,shaping='none; QoS and inventory are settled once at the endpoint',
        task_aggregation='task-weighted late fraction; not the legacy per-type nonlinear SA cost',
        training_scope='S5d/S6 must attach the final combined reward to the last real transition; no extra step')


class TeamRewardLedger:
    """Explicit per-episode accumulator, with atomic validation before append.

    Consume rows from get_last_energy_rows after a successful advance. At T,
    finalize the existing environment and supply its episode report on the SAME
    append. No I/O is performed here. New episodes require a new ledger/run_id.
    """
    def __init__(self, scenario, config, *, run_id):
        self._spec = reward_spec(scenario,config)
        self._run_id = _run_id(run_id)
        self._cursor = 0
        self._raw = dict.fromkeys(PERIODIC,0.)
        self._costs = dict.fromkeys(self._spec['components'],0.)
        self._count = 0
        self._last = self._terminal = None

    def get_spec(self):
        return deepcopy(self._spec)

    def get_last_reward(self):
        return deepcopy(self._last)

    def summary(self):
        cost = math.fsum(self._costs.values())
        return dict(schema_version=REWARD_VERSION,run_id=self._run_id,purpose=self._spec['purpose'],
            final_weights=self._spec['final_weights'],approved_for_formal_experiments=self._spec['approved_for_formal_experiments'],config_sha256=self._spec['config_sha256'],
            covered_end_seconds=self._cursor,end_seconds=self._spec['end_seconds'],
            complete=self._terminal is not None,step_count=self._count,periodic_raw=deepcopy(self._raw),
            weighted_costs=deepcopy(self._costs),total_reward=-cost,
            agent_rewards={a:-cost for a in AGENTS},terminal=deepcopy(self._terminal))

    def _component(self, name, raw, *, denominator=None):
        definition = self._spec['components'][name]
        den = definition['normalizer'] if denominator is None else denominator
        normalized = None if raw is None else (0. if den==0 else _number(raw/den,name+' normalized'))
        weighted = 0. if normalized is None else _number(normalized*definition['weight'],name+' weighted')
        return dict(raw=raw,raw_unit=definition['raw_unit'],normalizer=den,normalized=normalized,
            weight=definition['weight'],weighted_cost=weighted,phase=definition['phase'],available=raw is not None)

    @staticmethod
    def _match(actual, expected, name):
        if not math.isclose(_number(actual,name),expected,rel_tol=1e-10,abs_tol=1e-9):
            raise ValueError(name+' disagrees with the consumed execution records')

    def _terminal_components(self, report, totals):
        spec = self._spec
        end = spec['end_seconds']
        if (report['schema_version'] != EPISODE_REPORT_VERSION or report['run_id'] != self._run_id
                or report['physical_run_complete'] is not True or report['observation_end_seconds'] != end
                or report['cohort_seconds'] != spec['cohort_seconds']):
            raise ValueError('terminal report must match this run, cohort and full window')
        energy = report['energy']
        if (energy['run_id'] != self._run_id or energy['valid'] is not True or energy['complete'] is not True
                or energy['start_seconds'] != 0 or energy['end_seconds'] != end
                or energy['covered_end_seconds'] != end or energy['covered_duration_seconds'] != end
                or energy['grid_purchase_cost']['currency'] != spec['currency']):
            raise ValueError('terminal energy report must be complete, valid and in the same currency')
        self._match(energy['grid_purchase_cost']['amount'],totals['grid_cost'],'grid cost')
        self._match(energy['grid_carbon_kg'],totals['grid_carbon'],'grid carbon')
        if energy.get('joint_grid_control'):
            if energy['grid_tracking']['duration_seconds'] != end:
                raise ValueError('grid tracking duration disagrees with the full window')
            self._match(energy['grid_tracking']['mean_absolute_error_kw'],totals['grid_tracking']/end,'grid MAE')
        initial,final = energy['battery']['initial_energy_kwh'],energy['battery']['final_energy_kwh']
        self._match(initial,spec['initial_energy_kwh'],'initial energy')
        condition = terminal_energy_condition(initial,final)
        if report['terminal_energy'] != condition:
            raise ValueError('terminal energy condition disagrees with the existing evaluator')
        cohort,background = report['cohort_qos'],report['background']
        if ([cohort['cohort_start'],cohort['cohort_end']] != spec['cohort_seconds']
                or cohort['observed_until'] != end or background['observation_end_seconds'] != end
                or background['excluded_cohort_seconds'] != spec['cohort_seconds']):
            raise ValueError('task populations must match the declared cohort/window')
        c = {k:_seconds(cohort[k],'cohort '+k) for k in ('total','on_time','late','pending')}
        if c['pending'] or c['total'] != c['on_time']+c['late']:
            raise ValueError('cohort QoS must be resolved and internally consistent')
        b = {k:_seconds(background[k],'background '+k) for k in ('total','completed_on_time',
             'completed_late','overdue_unfinished','not_due_unfinished','deadline_violation_count','unfinished_count')}
        if (b['total'] != sum(b[k] for k in ('completed_on_time','completed_late','overdue_unfinished','not_due_unfinished'))
                or b['deadline_violation_count'] != b['completed_late']+b['overdue_unfinished']
                or b['unfinished_count'] != b['overdue_unfinished']+b['not_due_unfinished']):
            raise ValueError('background task counts are inconsistent')
        effective_debt = 0. if condition['condition_met'] else condition['shortfall_kwh']
        terms = dict(cohort_qos=self._component('cohort_qos',c['late'],denominator=c['total']),
            background_qos=self._component('background_qos',b['deadline_violation_count'],denominator=b['total']),
            terminal_energy=self._component('terminal_energy',effective_debt),
            not_due_backlog=self._component('not_due_backlog',b['not_due_unfinished']))
        details = dict(energy_condition=deepcopy(condition),cohort_counts=c,background_counts=b,
            cohort_has_samples=c['total']>0,background_has_samples=b['total']>0,
            cohort_formal_final=cohort['final'],effective_energy_shortfall_kwh=effective_debt,
            task_aggregation=spec['task_aggregation'])
        return terms,details

    def append(self, rows, *, terminal_report=None):
        if self._terminal is not None:
            raise RuntimeError('reward window already settled; use summary/get_last_reward for reads')
        rows = tuple(rows)
        if not rows:
            raise ValueError('a reward step requires new executed intervals')
        cursor = self._cursor
        amounts = {name:[] for name in PERIODIC}
        for row in rows:
            start,end = _seconds(row['start_seconds'],'start'),_seconds(row['end_seconds'],'end')
            if row['run_id'] != self._run_id or start != cursor or not start < end <= self._spec['end_seconds']:
                raise ValueError('reward intervals must be contiguous, ordered and from this run')
            grid = _number(row['power_kw']['grid_import'],'actual grid import',True)
            target = _number(row['signals']['grid_target_kw'],'grid target',True)
            if row['grid_purchase_cost']['currency'] != self._spec['currency']:
                raise ValueError('reward cost currency must match the scenario')
            amounts['grid_tracking'].append(_number(abs(grid-target)*(end-start),'grid error integral',True))
            amounts['grid_cost'].append(_number(row['grid_purchase_cost']['amount'],'grid cost'))
            amounts['grid_carbon'].append(_number(row['grid_carbon_kg'],'grid carbon',True))
            cursor = end
        terminal = cursor == self._spec['end_seconds']
        if terminal != (terminal_report is not None):
            raise ValueError('supply a terminal report exactly on the final physical reward step')
        raw = {name:_number(math.fsum(values),name+' step total') for name,values in amounts.items()}
        totals = {name:_number(math.fsum([self._raw[name],raw[name]]),name+' cumulative') for name in PERIODIC}
        components = {name:self._component(name,raw.get(name)) for name in self._spec['components']}
        details = None
        if terminal:
            terms,details = self._terminal_components(terminal_report,totals)
            components.update(terms)
        costs = {name:_number(math.fsum([self._costs[name],c['weighted_cost']]),name+' cumulative cost')
                 for name,c in components.items()}
        reward = -_number(math.fsum(c['weighted_cost'] for c in components.values()),'team reward')
        _number(math.fsum(costs.values()),'total cost')
        record = dict(schema_version=REWARD_VERSION,run_id=self._run_id,purpose=self._spec['purpose'],
            final_weights=self._spec['final_weights'],approved_for_formal_experiments=self._spec['approved_for_formal_experiments'],config_sha256=self._spec['config_sha256'],
            step_index=self._count,start_seconds=self._cursor,end_seconds=cursor,terminal_settlement=terminal,
            components=components,team_reward=reward,agent_rewards={a:reward for a in AGENTS},
            terminal=details)
        # Only commit after every row, terminal consistency and arithmetic check.
        self._cursor,self._raw,self._costs = cursor,totals,costs
        self._count += 1
        self._last,self._terminal = deepcopy(record),deepcopy(details)
        return record
