"""Deadline accounting, censoring, cohort consistency and replay boundaries."""
from io import StringIO
import json
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from peacsim.calculate_qos_cost import calculate_delay_prob, calculate_qos_cost_sa
from peacsim.extract_qos_metrics import write_qos_summary_df
from peacsim.parsing.experiment_config_reader import ExperimentConfigReader
from peacsim.qos_evaluation import (
    evaluate_qos, final_late_rates, constraint_results, prepare_qos_evaluation,
)


def table(arrivals, finishes, types=None):
    size = len(arrivals)
    return pd.DataFrame(dict(job_id=range(size), job_type_id=types if types is not None else [0]*size,
                             arrival_time=arrivals, start_time=arrivals, end_time=finishes,
                             min_execution_time=[10.]*size, qos_constraint=[1.]*size))


def evaluate(jobs, observed, end=25, start=0, count=1):
    return evaluate_qos(jobs, count, [1.]*count, [10.]*count, observed, start, end)


def test_agreed_100_task_example_and_fixed_denominator():
    jobs = table([0]*90+[20]*10, [10]*70+[-1]*30)
    _, interim = evaluate(jobs,25)
    row = interim.iloc[0]
    assert (row.total,row.on_time,row.late,row.pending) == (100,70,20,10)
    assert (row.late_rate_lower,row.late_rate_upper) == (.2,.3)
    assert np.isnan(row.late_rate)
    with pytest.raises(ValueError,match='pending'):
        final_late_rates(interim)
    # The original 20 late jobs eventually complete; they must stay late.
    jobs.loc[70:89,'end_time'] = 30
    jobs.loc[90:95,'end_time'] = 35
    jobs.loc[96:99,'end_time'] = 45
    _, final = evaluate(jobs,50)
    row = final.iloc[0]
    assert (row.total,row.on_time,row.late,row.pending) == (100,76,24,0)
    assert final_late_rates(final) == [.24]


@pytest.mark.parametrize('finish,observed,state',[
    (20,20,'on_time'), (21,21,'late'), (-1,20,'late'),
    (-1,19,'pending'), (19,18,'pending'), (21,20,'late'), (10,25,'on_time'),
])
def test_deadline_boundaries_and_no_future_information(finish,observed,state):
    tasks,_ = evaluate(table([0],[finish]),observed)
    assert tasks.qos_status.tolist() == [state]
    assert tasks.deadline.tolist() == [20]


@pytest.mark.parametrize('finishes,expected',[
    ([10,30],.5), ([30],1.), ([10],0.), ([-1,-1],1.), ([10,10,30,30],.5),
])
def test_exact_empirical_rate(finishes,expected):
    jobs=table([1]*len(finishes),finishes)
    assert calculate_delay_prob(1,1,jobs,[1],[10]) == [expected]


def test_fixed_cohort_includes_initial_arrivals_and_excludes_boundary_and_warmup():
    jobs=table([0,1,10,25,26],[10,12,19,40,-1])
    tasks,summary=evaluate(jobs,60)
    assert tasks.job_id.tolist() == [0,1,2]
    assert summary.total.tolist() == [3]
    tasks,summary=evaluate(jobs,60,start=1)
    assert tasks.job_id.tolist() == [1,2]
    assert summary.total.tolist() == [2]


def test_task_types_have_independent_rates():
    jobs=table([1]*4,[10,30,10,10],types=[0,0,1,1])
    _,summary=evaluate(jobs,60,count=2)
    assert final_late_rates(summary) == [.5,0.]


@pytest.mark.parametrize('jobs,count',[(table([],[]),1),(table([0],[10]),2)])
def test_empty_samples_are_unassessed_not_perfect(jobs,count):
    _,summary=evaluate(jobs,60,count=count)
    missing=summary.loc[summary.total==0]
    assert missing.late_rate.isna().all()
    assert (missing.evaluation_status=='no_samples').all()
    _,ratio=constraint_results(summary,.1)
    assert np.isnan(ratio)
    with pytest.raises(ValueError,match='no cohort samples'):
        final_late_rates(summary)


def test_exact_allowed_late_rate_is_feasible_despite_float_rounding():
    _,summary=evaluate(table([0]*10,[10]*9+[30]),60)
    flags,ratio=constraint_results(summary,1-.9)
    assert not flags.iloc[0]
    assert ratio == 0


def test_report_and_objective_use_identical_counts_and_original_softplus(tmp_path):
    jobs=table([1,1,30],[10,-1,40])  # Last task is background traffic, outside the cohort.
    sim=NS(_job_table=jobs,_simulation_duration=60,_qos_cohort_start=0,_qos_cohort_end=25,
           _job_type_count=1,_job_config=NS(all_job_qos_constraints={0:1.},all_min_execution_time={0:10.}),
           _output_dir=str(tmp_path),qos_violation_percentile_cutoff=.9)
    flags,ratio=write_qos_summary_df(sim)
    assert flags.iloc[0] and ratio==1
    report=pd.read_csv(tmp_path/'qos_summary.csv')
    assert report.total.tolist()==[2]
    assert report.late_rate.tolist()==[.5]
    assert set(pd.read_csv(tmp_path/'qos_task_status.csv').qos_status)=={'on_time','late'}
    config=NS(beta=2.,rho=5.,qos_constraint=.1)
    cost,each=calculate_qos_cost_sa(60/3600,1,jobs,[1],[10],config,True,cohort_end=25)
    expected=np.log(1+np.exp(5*(.5-.1)))
    assert each==pytest.approx([expected])
    assert cost==pytest.approx(2*expected)
    assert json.loads((tmp_path/'qos_evaluation.json').read_text())['final']


def test_pending_report_has_bounds_but_cannot_enter_objective(tmp_path):
    jobs=table([1,20],[10,-1])
    sim=NS(_job_table=jobs,_simulation_duration=25,_qos_cohort_start=0,_qos_cohort_end=25,
           _job_type_count=1,_job_config=NS(all_job_qos_constraints={0:1.},all_min_execution_time={0:10.}),
           _output_dir=str(tmp_path),qos_violation_percentile_cutoff=.9)
    flags,ratio=write_qos_summary_df(sim)
    assert pd.isna(flags.iloc[0]) and np.isnan(ratio)
    assert sim.qos_summary.late_rate_lower.tolist()==[0.]
    assert sim.qos_summary.late_rate_upper.tolist()==[.5]
    with pytest.raises(ValueError,match='pending'):
        calculate_qos_cost_sa(25/3600,1,jobs,[1],[10],NS(beta=2,rho=5,qos_constraint=.1))


def experiment(extra=''):
    return ExperimentConfigReader(StringIO('''[system]
simulation_duration=3600
time_granularity=1
random_seed=20
server_count=2
idle_watts=50
utilization=0.6
[iso]
normalize_iso_signal=True
randomize_iso_start=False
iso_signal_granularity=4
iso_file_path=../../data/ISO24h.csv
'''+extra))


def test_observation_window_is_policy_independent_and_saved_replay_is_idempotent(tmp_path):
    exp=experiment()
    jobs=NS(job_type_count=2,all_min_execution_time={0:25,1:44},all_job_qos_constraints={0:3,1:5})
    assert prepare_qos_evaluation(exp,jobs)==7200
    assert (exp.qos_cohort_start,exp.qos_cohort_end)==(0,3600)
    assert prepare_qos_evaluation(exp,jobs)==7200
    path=tmp_path/'中文配置.ini'
    exp.write_experiment_config(path)
    restored=ExperimentConfigReader(path)
    assert prepare_qos_evaluation(restored,jobs)==7200
    assert (restored.qos_cohort_start,restored.qos_cohort_end)==(0,3600)
    # CSV/JSON serialization must also be identical before and after config replay.
    assert type(exp.qos_cohort_start) is type(restored.qos_cohort_start) is float
    assert type(exp.qos_cohort_end) is type(restored.qos_cohort_end) is float


@pytest.mark.parametrize('start,end',[(-1,100),(100,100),(0,3601)])
def test_invalid_window_is_rejected_before_a_run(start,end):
    exp=experiment(f'[qos_evaluation]\ncohort_start={start}\ncohort_end={end}\n')
    with pytest.raises(ValueError,match='cohort'):
        prepare_qos_evaluation(exp,NS(job_type_count=1,all_min_execution_time={0:10},all_job_qos_constraints={0:1}))


def test_native_run_retains_background_arrivals_and_resolves_the_cohort(tmp_path):
    from peacsim.aqa_runtimepolicy import AQARuntimePolicy
    from peacsim.create_tables import init_node_table, generate_job_table
    from peacsim.run_simulator import init_output
    from peacsim.simulator import Simulator

    signal=tmp_path/'signal.csv';signal.write_text('0\n'*7)
    weights=tmp_path/'weights.csv';weights.write_text('job_type_id,weights\n0,1\n')
    exp=NS(server_count=1,idle_power=50.,time_granularity=2,random_seed=20,simulation_duration=24,
           qos_cohort_start=0,qos_cohort_end=8,utilization=.6,normalize_iso_signal=False,
           iso_signal_granularity=4,iso_file_path=str(signal),randomize_iso_start=False,iso_signal_start_hour=0)
    jobs=NS(job_type_count=1,all_jobs={'job':0},all_max_job_power={0:200.},all_min_job_power={0:100.},
            all_min_execution_time={0:4.},all_max_execution_time={0:8.},all_job_size={0:1},all_job_qos_constraints={0:2.})
    pc=NS(runtime_policy_name='AQA',P=200.,R=50.,policy_parameters={'weights_path':str(weights),'node_count_control':False})
    jt=generate_job_table(3,[0,0,0],[0,6,10],jobs);nt=init_node_table(exp)
    policy=AQARuntimePolicy(pc,exp,jobs,nt,jt)
    sim=Simulator(jt,nt,policy,exp,jobs,NS(),output_dir=str(tmp_path)+'/')
    init_output(str(tmp_path)+'/',exp,pc);sim.run()
    assert pd.read_csv(tmp_path/'power_trace.csv').time.tolist()==list(range(0,25,2))
    assert sim.qos_summary.total.tolist()==[2]
    assert sim.qos_summary.pending.tolist()==[0]
    assert sim.qos_summary.on_time.tolist()==[2]
    assert sim._job_table.loc[2,'start_time']>=10
    initial=pd.read_csv(tmp_path/'qos_summary_at_cohort_end.csv')
    assert initial.pending.tolist()==[1]
    a=json.loads((tmp_path/'qos_evaluation_at_cohort_end.json').read_text())
    b=json.loads((tmp_path/'qos_evaluation.json').read_text())
    assert a['cohort_sha256']==b['cohort_sha256']
