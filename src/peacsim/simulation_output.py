"""Initialize the three legacy CSV files before a new simulation starts.

The caller owns directory creation, experiment isolation and call timing.
Opening with 'w' retains the old initialization semantics: never call this to
resume a running experiment. This module does not build, advance or reset one.
"""
import os


def init_output(output_dir, experiment_config, policy_config):
    """Write unchanged headers/initial P,R; accept directories with or without '/'."""
    with open(os.path.join(output_dir, 'power_trace.csv'), 'w') as f:
        f.write('time,QoSemergency,numNeededForTracking,waitingSum,target,realSum,trackingError,QoSJobSum,standbyPower')
        for i in range(experiment_config.server_count):
            f.write(',client%d' % (i + 1))
        f.write(',estimateSum')
        for i in range(experiment_config.server_count):
            f.write(',state%d' % (i + 1))
        f.write('\n')
    with open(os.path.join(output_dir, 'PRtable.csv'), 'w') as f:
        f.write('phase,P,R\n')
        f.write('%d,%d,%d\n' % (0, policy_config.P, policy_config.R))
    with open(os.path.join(output_dir, 'job_table.csv'), 'w') as f:
        f.write('job_id,job_type_id,arrival_time,start_time,end_time,estimate_finished,update_time,realtime_qos,'
                'min_execution_time,qos_constraint,job_size\n')
