"""Focused regression tests. Strict xfails document unresolved model defects."""
import pathlib, sys
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/peacsim'))
from peacsim import simulated_annealing as sa
from peacsim.simulator import Simulator
from peacsim.calculate_qos_cost import calculate_delay_prob
from peacsim.aqa_runtimepolicy import AQARuntimePolicy
from peacsim.create_tables import init_node_table

@pytest.mark.parametrize('costs, expected_index', [([1,10,10,10,10,10],-1),([10,1,10,10,10,10],0)])
def test_optimizer_returns_best_pair_even_without_improvement(monkeypatch,costs,expected_index):
    cfg=NS(p_low=.9,p_high=1.1,r_low=.55,r_high=.8,w_low=.2,w_high=.3,
           p_init=1.,r_init=.8,temperature=1.,cooling_rate=.95,p_step=.01,r_step=.01,w_step=.01,
           max_iter=5,smart_weight=True,program_type='RSR')
    candidates=[]
    def evaluate(params,*args,**kwargs):
        cost=costs[len(candidates)]; candidates.append(list(params))
        return cost,0.,0.,0.,0.,0.,[0.]*4
    monkeypatch.setattr(sa,'sa_evaluate',evaluate)
    monkeypatch.setattr(sa.wandb,'log',lambda *a,**k:None)
    monkeypatch.setattr(np.random,'rand',lambda: .999)
    best,index=sa.simulated_annealing_optimize(cfg,NS(runtime_policy_name='AQA'),NS(),NS(),NS(job_type_count=4),'unused')
    assert index==expected_index
    assert best==candidates[expected_index+1]
    assert len(candidates)==6

@pytest.mark.xfail(strict=True,reason='Unfixed: a multi-node job completes when any node reaches 1')
def test_multinode_requires_all_peers():
    s=Simulator.__new__(Simulator)
    n=init_node_table(NS(server_count=2,idle_power=120))
    n[1]=0; n[2]=0; n[5]=[1,.5]; s._node_table=n
    nodes,jobs=s.check_all_other_servers_finished()
    assert len(nodes)==0 and len(jobs)==0

def test_two_jobs_one_violation_has_probability_half():
    jobs=pd.DataFrame({'job_type_id':[0,0],'arrival_time':[1,1],'end_time':[11,31]})
    assert calculate_delay_prob(1,1,jobs,[1],[10])==[.5]

def test_lifting_cap_restores_commanded_power():
    p=AQARuntimePolicy.__new__(AQARuntimePolicy)
    n=init_node_table(NS(server_count=1,idle_power=120))
    n[1]=0; n[2]=0; n[3]=150; n[4]=200; n[8]=100; n[9]=200; n[11]=-1
    p._node_table=n
    p.apply_power_cap(200)
    assert n[11,0]==200
    assert n[3,0]==150  # The next simulator update applies the command.
