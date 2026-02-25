import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.peacsim.notebooks.plotting_helpers import get_job_table_df
from peacsim.parsing import job_profile_reader as job_parser

def calculate_qos(job_table):
    """
    Calculate QoS for started jobs using the selected logic.
    """
    started_jobs = job_table[
        (job_table['end_time'] > 0) &  # Ensure jobs have ended
        (job_table['arrival_time'] <= 18000)  # Filter jobs within the simulation duration
    ]
    if started_jobs.empty:
        return []

    # Calculate QoS for each job
    qos = (started_jobs['end_time'] - started_jobs['arrival_time'] - started_jobs['min_execution_time']) / started_jobs['min_execution_time']
    return qos

def plot_cdf_comparison(exp_names, job_config_path, output_dir_base, job_config_file):
    """
    Plot CDFs for QoS values calculated for multiple experiments.
    """
    job_config = job_parser.JobProfileReader(job_config_path + job_config_file)
    qos_constraints = job_config.all_job_qos_constraints

    # Initialize data storage
    qos_data_dict = {exp_name: [] for exp_name in exp_names}

    # Process each experiment
    for exp_name in exp_names:
        output_dir = f"{output_dir_base}/{exp_name}"
        job_table = get_job_table_df(output_dir)

        for job_type in range(job_config.job_type_count):
            current_job_type = job_table[job_table['job_type_id'] == job_type]
            qos_values = calculate_qos(current_job_type)
            qos_data_dict[exp_name].append(qos_values)

    # Plot CDFs in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True)
    axes = axes.flatten()
    # Use simple, distinct colors and line styles for clarity
    colors = ['darkblue', 'darkorange', 'tab:green', 'tab:red'][:len(exp_names)]
    line_styles = ['-', '-.', '--'][:len(exp_names)]
    # Friendly display names for legend in order of exp_names
    display_names = []
    if len(exp_names) >= 1:
        display_names.append("ProportionalCap")
    if len(exp_names) >= 2:
        display_names.append("PriorityCap")
    if len(exp_names) >= 3:
        display_names.append("AQA")

    for i, ax in enumerate(axes):
        if i < job_config.job_type_count:
            for j, exp_name in enumerate(exp_names):
                label_name = display_names[j] if j < len(display_names) else exp_name
                sns.ecdfplot(qos_data_dict[exp_name][i], color=colors[j], ax=ax,
                             label=label_name, linestyle=line_styles[j], linewidth=2)

            # Add QoS threshold and 90th percentile lines
            ax.axhline(0.9, color='black', linestyle='dotted', linewidth=2, label='90th %ile')
            ax.axvline(qos_constraints[i], color='red', linestyle='--', linewidth=2, label='Q_th')

            # Set labels (y-label only on leftmost column)
            if i % 2 == 0:
                ax.set_ylabel("CDF", fontsize=12)
            else:
                ax.set_ylabel("")
            ax.set_xlabel(job_config.all_jobs[i], fontsize=12)
            ax.set_xlim(0, 5.2)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Create legend: first row policies, second row reference lines
    policy_handles = [
        plt.Line2D([0], [0], color=colors[j], linewidth=2, linestyle=line_styles[j],
                   label=(display_names[j] if j < len(display_names) else exp_name))
        for j, exp_name in enumerate(exp_names)
    ]
    ref_handles = [
        plt.Line2D([0], [0], color='black', linestyle='dotted', linewidth=2, label="90th %ile"),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label="Q_th"),
    ]

    # Two stacked legends for deterministic layout
    ncols_policies = max(1, len(policy_handles))
    fig.legend(handles=policy_handles, loc='upper center', fontsize=12,
               ncol=ncols_policies, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.legend(handles=ref_handles, loc='upper center', fontsize=12,
               ncol=len(ref_handles), frameon=True, bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('plots/qos_cdf_comparison.pdf', bbox_inches='tight')
    plt.show()

# Example usage
exp_names = [
    # "dr_EDR_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_traditional_signal_qos0.9_20250916164642",
    # "dr_EDR_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_traditional_signal_qos0.9_20250916135804",
    "sim_policy=proportional-cap_dr_EDR_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_traditional_signal_qos0.9_20250917170139",
    "sim_policy=priority-cap_dr_EDR_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_traditional_signal_qos0.9_20250917170346",
    "dr_EDR_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_traditional_signal_qos0.9_20250908134942"
]
job_config_path = "../../../configs/workload/"
output_dir_base = "../output/simulation"
job_config_file = "W2-short-qos5_4.5_4_3.5.ini"

plot_cdf_comparison(exp_names, job_config_path, output_dir_base, job_config_file)
