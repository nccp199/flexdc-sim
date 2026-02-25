from plotting_helpers import get_job_table_df, get_power_trace_df
from peacsim.parsing import job_profile_reader as job_parser
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd

optimization_dir = "opt_flex-resource_probabilities=0.8,0.8,0.8,0.8_SA_inf_low_util_RSR_1000server_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_dynamic_signal_20250915100931/"
exp_name = "flex_resource_prob=0.8_dr_RSR_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_dynamic_signal_qos0.9_20250917111611"

weigts_file_path = "../output/optimization/" + optimization_dir + "/opt_weights.csv"   ### mock_weights.csv


output_dir = "../output/simulation/" + exp_name

job_table = get_job_table_df(output_dir)
power_df = get_power_trace_df(output_dir)


 # Replace with the actual file path
weights_df = pd.read_csv(weigts_file_path)

# Convert to dictionary
job_type_weights = weights_df.set_index('job_type_id')['weights'].to_dict()


state_columns = [col for col in power_df.columns if col.startswith('state')]

# Step 2: Calculate job type distributions over time
# Melt the DataFrame to make it long-format for easier processing
melted = power_df.melt(id_vars='time', value_vars=state_columns, var_name='server', value_name='job_type')

# Exclude inactive servers
active_jobs = melted[melted['job_type'] != -1]

# Count job types per time
active_job_type_counts = active_jobs.groupby(['time', 'job_type']).size().unstack(fill_value=0)

# Step 3: Normalize counts to get ratios
active_job_type_ratios = active_job_type_counts.div(active_job_type_counts.sum(axis=1), axis=0)


waiting_jobs = pd.read_csv(output_dir + "/waiting_jobs_by_job_type.csv")
waiting_jobs.fillna(0, inplace=True)
waiting_jobs.index += 1
active_job_type_counts.columns = waiting_jobs.columns

num_servers = 1000
weights = np.asarray(weights_df['weights'].values)
guaranteed_resources = (num_servers * weights)  ### TODO ROUND OR NOT
guaranteed_resources = np.floor(guaranteed_resources).astype(int)
guaranteed_resources.reshape(1, len(guaranteed_resources))
waiting_and_active_jobs_sum = active_job_type_counts.add(waiting_jobs, fill_value=0)

guaranteed_resources_df = pd.DataFrame([guaranteed_resources], columns=waiting_and_active_jobs_sum.columns)

# Compute demand and required guaranteed allocation per time and job type
demand = active_job_type_counts.add(waiting_jobs, fill_value=0)
required = pd.DataFrame(
    np.minimum(guaranteed_resources_df.iloc[0].values, demand.values),
    index=demand.index, columns=demand.columns
)

# Success when active >= min(A_j, demand)
success = active_job_type_counts.ge(required)

# Ratio of time steps meeting the guarantee
true_ratio = success.mean(axis=0)
print(true_ratio)

shortfall = np.maximum(0, 0.8 - true_ratio)

mae = np.mean(shortfall)
print(f"MAE: {mae}")