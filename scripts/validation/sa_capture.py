"""Capture native SA evaluations without changing its decisions or random seeds."""
import json
from pathlib import Path
import sys

assert sys.argv[1] == '--repo'
repo = Path(sys.argv[2]).resolve()
path = repo / 'src/peacsim/simulated_annealing.py'
sys.argv = [str(path), *sys.argv[3:]]
sys.path[:0] = [str(repo / 'src'), str(repo / 'src/peacsim')]
source = path.read_text(encoding='utf-8')
wrapper = '''
_original_sa_evaluate = sa_evaluate
def sa_evaluate(params, simulated_annealing_config, policy_config, experiment_config,
                cluster_config, job_config, output_dir, iteration):
    import hashlib
    candidate_dir = os.path.join(output_dir, 'evaluations', '%03d' % iteration) + '/'
    os.makedirs(candidate_dir, exist_ok=True)
    result = _original_sa_evaluate(params, simulated_annealing_config, policy_config,
        experiment_config, cluster_config, job_config, candidate_dir, iteration)
    entry = dict(iteration=iteration, params=list(params), result=result,
        workload_seed=workload_seed, scheduling_seed=experiment_config.random_seed,
        workload_sha256=hashlib.sha256(job_table_.tobytes()).hexdigest())
    with open(os.path.join(output_dir, 'evaluations.jsonl'), 'a') as f:
        f.write(json.dumps(entry, default=lambda x: x.item() if hasattr(x, 'item') else list(x)) + '\\n')
    return result
'''
needle = 'if __name__ == "__main__":'
assert source.count(needle) == 1
exec(compile(source.replace(needle, wrapper + '\n' + needle), str(path), 'exec'),
     {'__name__': '__main__', '__file__': str(path)})
