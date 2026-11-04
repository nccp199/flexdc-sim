"""Mac acceptance: fixed-cohort QoS, identical physical trajectories and replay."""
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import importlib.metadata
import platform
import argparse

import numpy as np
import pandas as pd
import psutil

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT = args.repo.resolve()
BEFORE = args.baseline.resolve()
TOOLS = Path(__file__).resolve().parent
OUT = args.output.resolve()
OUT.mkdir(exist_ok=False)
TOKEN = uuid.uuid4().hex[:10]
FILES = ['PRtable.csv','QoSDegAvg.csv','cost_power_rsr.csv','job_table.csv','power_trace.csv',
         'qos_summary.csv','tracking_error_90th.csv','qos_task_status.csv','qos_evaluation.json',
         'qos_summary_at_cohort_end.csv','qos_task_status_at_cohort_end.csv','qos_evaluation_at_cohort_end.json']
results = {'cases':{},'commands':{},'checks':{},'optimization':{},'environment':{
    'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n)
    for n in ['numpy','pandas','scipy','pytest','wandb','tqdm']}}}


def checkpoint():
    (OUT/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


def run(repo,label,args):
    env=os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(repo/'src'),str(repo/'src/peacsim'),str(repo)]),
               WANDB_MODE='disabled',MPLCONFIGDIR=str(OUT/'mpl'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,*map(str,args)]
    start=time.perf_counter();peak=0
    print('Running '+label,flush=True)
    log=OUT/(label+'.log')
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.Popen(command,cwd=repo/'src/peacsim',env=env,stdout=f,stderr=subprocess.STDOUT)
        process=psutil.Process(p.pid)
        while p.poll() is None:
            try:
                peak=max(peak,process.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            time.sleep(.05)
    results['commands'][label]=dict(command=command,cwd=str(repo/'src/peacsim'),
                                   seconds=time.perf_counter()-start,peak_root_RSS_MiB=peak/2**20,exit_code=p.returncode)
    checkpoint()
    if p.returncode:
        raise RuntimeError(label+' failed: '+str(log))
    return log.read_text(encoding='utf-8',errors='replace')


def path_from_log(repo,text):
    return (repo/'src/peacsim'/re.findall(r'Results saved in:\s*([^\r\n]+)',text)[-1].strip()).resolve()


def same(a,b,names):
    checks={name:(a/name).read_bytes()==(b/name).read_bytes() for name in names}
    assert all(checks.values()),checks
    return checks


def inspect_case(path):
    jobs=pd.read_csv(path/'job_table.csv')
    summary=pd.read_csv(path/'qos_summary.csv')
    tasks=pd.read_csv(path/'qos_task_status.csv')
    close=pd.read_csv(path/'qos_summary_at_cohort_end.csv')
    meta=json.loads((path/'qos_evaluation.json').read_text())
    close_meta=json.loads((path/'qos_evaluation_at_cohort_end.json').read_text())
    start,end,observed=meta['cohort_start'],meta['cohort_end'],meta['observed_until']
    cohort=jobs.loc[(jobs.arrival_time>=start)&(jobs.arrival_time<end)].copy()
    deadline=cohort.arrival_time+(1+cohort.qos_constraint)*cohort.min_execution_time
    success=(cohort.end_time>=0)&(cohort.end_time<=observed)&(cohort.end_time<=deadline)
    late=((cohort.end_time>=0)&(cohort.end_time<=observed)&(cohort.end_time>deadline))|((cohort.end_time<0)&(observed>=deadline))
    expected=np.where(success,'on_time',np.where(late,'late','pending'))
    assert tasks.job_id.tolist()==cohort.job_id.tolist()
    assert tasks.qos_status.tolist()==expected.tolist()
    assert not (expected=='pending').any()
    assert meta['cohort_sha256']==close_meta['cohort_sha256']
    assert meta['total']==close_meta['total']==len(cohort)
    for _,row in summary.iterrows():
        selected=tasks.loc[tasks.job_type_id==row.job_type_id]
        assert row.total==len(selected)
        assert (row.on_time,row.late,row.pending)==tuple(int((selected.qos_status==state).sum()) for state in ['on_time','late','pending'])
        assert np.isclose(row.late_rate,row.late/row.total)
    assert (jobs.arrival_time>=end).any(), 'normal traffic must continue after cohort close'
    power=pd.read_csv(path/'power_trace.csv',usecols=['time','target','realSum'])
    assert power.time.iloc[-1]==observed==7200
    energy=float(np.dot(power.realSum.to_numpy()[:-1],np.diff(power.time.to_numpy()))/3_600_000)
    return dict(path=str(path),cohort=meta,at_cohort_end=close_meta,by_type=summary.to_dict(orient='records'),
                energy_kWh=energy,background_task_count=int((jobs.arrival_time>=end).sum()),
                late_but_unfinished=int(((tasks.qos_status=='late')&~tasks.completed_as_of).sum()))


policy=configparser.ConfigParser();policy.read(TOOLS/'configs/policy.ini',encoding='utf-8-sig')
policy['parameters']['weights_path']=str(TOOLS/'configs/original_weights.csv')
policy_path=OUT/'policy.ini'
with policy_path.open('w') as f:policy.write(f)
common=['--cluster-config',TOOLS/'configs/cluster.ini','--job-config',TOOLS/'configs/jobs.ini']


def simulation(repo,label,exp,pol=policy_path,extra=()):
    text=run(repo,label,[repo/'src/peacsim/run_simulator.py','--experiment-config',exp,
        '--policy-config',pol,*common,'--output-dir',TOKEN+'_'+label,*extra])
    return path_from_log(repo,text)


for n in [50,1000]:
    exp=TOOLS/'configs'/f'exp{n}.ini'
    old=configparser.ConfigParser();old.read(exp,encoding='utf-8-sig');old['system']['simulation_duration']='7200'
    old_path=OUT/f'before_exp{n}.ini'
    with old_path.open('w') as f:old.write(f)
    before=simulation(BEFORE,f'before_{n}',old_path,extra=['--convert-from-normalized-PR'])
    after=simulation(ROOT,f'after_{n}',exp,extra=['--convert-from-normalized-PR'])
    repeat=simulation(ROOT,f'after_{n}_repeat',exp,extra=['--convert-from-normalized-PR'])
    results['checks'][f'physical_trajectory_{n}']=same(before,after,[f for f in FILES[:7] if f!='qos_summary.csv'])
    results['checks'][f'repeat_{n}']=same(after,repeat,FILES)
    results['cases'][f'after_{n}']=inspect_case(after)
    results['cases'][f'after_{n}_repeat']=inspect_case(repeat)
    results['cases'][f'before_{n}']={'path':str(before)}
    checkpoint()

label='after_sa5';prefix=TOKEN+'_'+label
run(ROOT,label,[TOOLS/'sa_capture.py','--repo',ROOT,'--experiment-config',TOOLS/'configs/exp50.ini',
    *common,'--gradient-config',TOOLS/'configs/SA5.ini','--policy-name','AQA','--output-dir',prefix])
folders=list((ROOT/'src/peacsim/output/optimization').glob(prefix+'*'))
assert len(folders)==1
sa=folders[0]
best=json.loads((sa/'best_result.json').read_text())
evaluations=[json.loads(line) for line in (sa/'evaluations.jsonl').read_text().splitlines()]
assert len(evaluations)==6
assert len({e['workload_sha256'] for e in evaluations})==1
cohort_hashes=[]
for evaluation in evaluations:
    case=inspect_case(sa/'evaluations'/f"{evaluation['iteration']:03d}")
    cohort_hashes.append(case['cohort']['cohort_sha256'])
    rates=np.asarray([row['late_rate'] for row in case['by_type']])
    penalty=2*np.log(1+np.exp(5*(rates-.1))).sum()
    assert np.isclose(penalty,evaluation['result'][2],rtol=1e-12,atol=1e-12)
    assert np.allclose(np.log(1+np.exp(5*(rates-.1))),evaluation['result'][6],rtol=1e-12,atol=1e-12)
    pr=pd.read_csv(sa/'evaluations'/f"{evaluation['iteration']:03d}"/'PRtable.csv')
    # Monetary term covers both hours, with the original rate and bidding formula.
    assert np.isclose(evaluation['result'][1],.1/1000*(float(pr.P.iloc[0])-float(pr.R.iloc[0]))*2)
assert len(set(cohort_hashes))==1
assert cohort_hashes[0]==results['cases']['after_50']['cohort']['cohort_sha256']
replay=simulation(ROOT,'after_best_replay',sa/'experiment_config.ini',sa/'optimal_policy.ini',
                  ['--scheduling-seed',str(best['scheduling_seed'])])
results['checks']['best_replay']=same(sa/'evaluations'/f"{best['evaluation_index']:03d}",replay,FILES)
results['checks']['all_sa_objectives_match_report']=True
results['checks']['same_cohort_across_normal_run_and_sa']=True
results['checks']['observation_cost_includes_both_hours']=True
results['cases']['after_best_replay']=inspect_case(replay)
results['optimization']={'path':str(sa),'best':best,'evaluations':evaluations}
run(ROOT,'pytest',['-m','pytest',ROOT/'tests',ROOT/'reports/reproduction_20261104/scripts/test_correctness.py',
                  '-q','-rsx','--junitxml='+str(OUT/'pytest.xml')])
results['status']='passed';checkpoint()
print('All QoS acceptance checks passed: '+str(OUT/'summary.json'),flush=True)
