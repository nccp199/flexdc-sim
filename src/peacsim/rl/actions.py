"""S5b: bounded sampled preferences -> physical requests, without a trainer.

The raw preference box is the probability space. Normalization is many-to-one:
common positive rescaling changes the sampled action but not its allocation.
Never replace a sampled vector with normalized weights or executed powers when
computing policy probability ratios. A future simplex sampler is a new schema.
"""
from dataclasses import asdict, dataclass

import numpy as np

from peacsim.control.allocation import normalize_task_weights
from peacsim.control.joint_action import JointAction
from peacsim.energy.facility import BatteryRequest
from peacsim.metrics.tracking import _number


ACTION_VERSION = 'flexdc-action-v1'


def action_spec(*, job_config, server_count, idle_power_w, environment):
    """Static bounds and true-dimension masks; no state-dependent padding."""
    k = job_config.job_type_count
    low = _number(server_count*idle_power_w/1000., 'minimum IT budget', True)
    high = _number(server_count*max(idle_power_w,*job_config.all_max_job_power.values())/1000.,
                   'maximum IT budget', True)
    battery = environment.battery
    fields = []
    for j in range(k):
        fields.append(dict(name='type_'+str(j)+'_preference', unit='1', bounds=[0.,1.],
                           meaning='nonnegative relative resource preference; normalized across configured types'))
    fields.append(dict(name='it_budget_fraction',unit='1',bounds=[0.,1.],
                       meaning='linear map from all-idle power to the declared cluster maximum'))
    return dict(schema_version=ACTION_VERSION,agent_order=['compute','storage'],task_type_order=list(range(k)),
                dimensions=dict(compute=k+1,storage=1),
                fields=dict(compute=fields,storage=[dict(name='battery_signed_fraction',unit='1',bounds=[-1.,1.],
                                                        meaning='positive discharge; negative charge; zero standby')]),
                active_dimensions=dict(compute=[k>1]*k+[high>low],
                                       storage=[battery.max_charge_power_kw>0 or battery.max_discharge_power_kw>0]),
                inactive_canonical_value=0.,physical_bounds=dict(it_min_kw=low,it_max_kw=high,
                    max_charge_kw=battery.max_charge_power_kw,max_discharge_kw=battery.max_discharge_power_kw),
                zero_preference_rule='uniform across configured task types; logged explicitly',
                queue_rule='reuse native adjust_weights each physical scheduling call; zero priority may defer a queue',
                holding_rule='reuse common control interval and original delayed node actuation',
                probability_contract=dict(action='original_policy_action',
                    mask='active_dimensions; omit statically inactive coordinates from log probability and entropy',
                    normalization='deterministic environment mapping; no density assigned to the normalized weights',
                    redundancy='K sampled preferences have K-1 physical allocation degrees of freedom; common-scale redundancy retained',
                    entropy='raw active sampled-coordinate entropy, not physical allocation entropy',
                    distribution_implementation='S6; no probability or optimizer is implemented in S5b'))


def _vector(value, fields, label):
    array = np.asarray(value,dtype=object)
    if array.ndim != 1 or len(array) != len(fields):
        raise ValueError(label+' action must have its declared one-dimensional shape')
    result = tuple(_number(v,label+' action') for v in array)
    for v,field in zip(result,fields):
        low,high = field['bounds']
        if not low <= v <= high:
            raise ValueError(label+' action is outside declared bounds; do not silently clip sampled actions')
    return result


@dataclass(frozen=True)
class DecodedActions:
    compute: tuple
    storage: tuple
    task_weights: tuple
    joint_action: JointAction
    compute_mask: tuple
    storage_mask: tuple
    zero_preference_fallback: bool

    def to_record(self):
        return dict(schema_version=ACTION_VERSION,
                    original_policy_action=dict(compute=list(self.compute),storage=list(self.storage)),
                    active_dimensions=dict(compute=list(self.compute_mask),storage=list(self.storage_mask)),
                    task_weights=list(self.task_weights),joint_action=asdict(self.joint_action),
                    zero_preference_fallback=self.zero_preference_fallback,
                    probability_action='original_policy_action; never substitute executed power')


def decode_actions(compute, storage, *, spec):
    """Copy inputs, validate the policy box, then build unexecuted physical commands."""
    if spec['schema_version'] != ACTION_VERSION:
        raise ValueError('unsupported action schema')
    a = _vector(compute,spec['fields']['compute'],'compute')
    b = _vector(storage,spec['fields']['storage'],'storage')
    k = len(spec['task_type_order'])
    weights = normalize_task_weights(a[:k],k,zero_uniform=True)
    limits = spec['physical_bounds']
    fraction = a[-1]
    budget = limits['it_min_kw']+(limits['it_max_kw']-limits['it_min_kw'])*fraction
    battery = BatteryRequest(charge_power_kw=max(0.,-b[0])*limits['max_charge_kw'],
                             discharge_power_kw=max(0.,b[0])*limits['max_discharge_kw'])
    return DecodedActions(a,b,weights,JointAction(budget,battery),
                          tuple(spec['active_dimensions']['compute']),tuple(spec['active_dimensions']['storage']),
                          max(a[:k])==0.)
