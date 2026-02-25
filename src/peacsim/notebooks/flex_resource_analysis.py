# Correct evaluation of FlexResource availability compliance
# Copy-paste and set the two paths below, then run.

from plotting_helpers import get_power_trace_df
import pandas as pd
import numpy as np

# --- SET THESE ---
optimization_dir = "opt_flex-resource_probabilities=0.8,0.8,0.8,0.8_SA_inf_low_util_RSR_1000server_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_dynamic_signal_20250915100931/"
exp_name = "999_dr_RSR_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_dynamic_signal_qos0.9_20250915193646/"
# -----------------

weigts_file_path = "../output/optimization/" + optimization_dir + "/opt_weights.csv"
output_dir = "../output/simulation/" + exp_name

# Load weights and outputs
weights_df = pd.read_csv(weigts_file_path)
power_df = get_power_trace_df(output_dir)
waiting_jobs = pd.read_csv(output_dir + "/waiting_jobs_by_job_type.csv").fillna(0)

# Identify server state columns and build active counts per job type and time
state_columns = [c for c in power_df.columns if c.startswith('state')]
melted = power_df.melt(id_vars='time', value_vars=state_columns, var_name='server', value_name='job_type')
active_jobs = melted[melted['job_type'] != -1]
active_job_type_counts = active_jobs.groupby(['time', 'job_type']).size().unstack(fill_value=0)

# Align columns: use waiting_jobs column order as the canonical order
active_job_type_counts = active_job_type_counts.reindex(columns=waiting_jobs.columns, fill_value=0)

# Compute N and A_j
num_servers = len(state_columns)
# Ensure weights align to columns by job_type_id
weights_map = weights_df.set_index('job_type_id')['weights'].to_dict()
weights_vector = np.array([weights_map.get(int(col), 0.0) for col in active_job_type_counts.columns])
guaranteed_resources = np.round(num_servers * weights_vector).astype(int)  # A_j = round(N * w_j)

# Demand and required guaranteed allocation
demand = active_job_type_counts.add(waiting_jobs, fill_value=0)
required = pd.DataFrame(
    np.minimum(guaranteed_resources.reshape(1, -1).repeat(len(demand), axis=0), demand.values),
    index=demand.index,
    columns=demand.columns
)

# Success when active >= min(A_j, demand)
success = active_job_type_counts.ge(required)

# Ratio of time steps meeting the guarantee per job type
true_ratio = success.mean(axis=0)

print("True ratio per job type:")
print(true_ratio)