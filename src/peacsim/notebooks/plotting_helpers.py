import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def get_power_trace_df(output_dir):
    power_df = pd.read_csv(output_dir + '/power_trace.csv').iloc[1:]
    state_columns = [col for col in power_df.columns if col.startswith('state')]

    power_df['active_servers'] = (power_df[state_columns] != -1).sum(axis=1)
    power_df['realSum_kw'] = power_df['realSum'] / 1000  # Convert to kw
    power_df['target_kw'] = power_df['target'] / 1000  # Convert to kw

    return power_df


def get_job_table_df(output_dir):
    job_table = pd.read_csv(output_dir + '/job_table.csv')
    return job_table


def plot_power_consumption(ax, power_df):
    ax.plot(power_df['time'], power_df['realSum_kw'], label='Power Consumption', linewidth=2.5, color='blue')
    ax.plot(power_df['time'], power_df['target_kw'], label='Power Target', color='red', linewidth=2, linestyle='--')
    ax.set_ylabel('Power (kW)', fontsize=16)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.6)
    ax.tick_params(axis='both', which='major', labelsize=14)


def plot_active_servers(ax, power_df):
    ax.plot(power_df['time'], power_df['active_servers'], label='Active Servers', color='green', linewidth=2)
    ax.set_ylabel('Active Servers', fontsize=16)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, linestyle='--', linewidth=0.6)
    ax.tick_params(axis='both', which='major', labelsize=14)


def plot_waiting_jobs(ax, power_df):
    ax.plot(power_df['time'], power_df['waitingSum'], label='Jobs in Queue', color='purple', linewidth=2)
    ax.set_ylabel('Waiting Jobs', fontsize=16)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.6)
    ax.tick_params(axis='both', which='major', labelsize=14)


def plot_real_time_qos(ax, power_df):
    ax.plot(power_df['time'], power_df['realtimeQoS'], label='Real-Time QoS', color='orange', linewidth=2)
    ax.set_ylabel('Real-Time QoS', fontsize=16)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.6)
    ax.tick_params(axis='both', which='major', labelsize=14)


def plot_power_trace(power_df, file_name_prefix, include_power=True, include_servers=True, include_jobs=True,
                     include_qos=True):
    # Determine the number of subplots based on parameters
    subplots = [include_power, include_servers, include_jobs, include_qos]
    num_subplots = sum(subplots)

    # Define figure and subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(22, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.0] * num_subplots})
    if num_subplots == 1:
        axes = [axes]  # Ensure axes is always a list

    # Plot each subplot if included
    idx = 0
    if include_power:
        plot_power_consumption(axes[idx], power_df)
        idx += 1
    if include_servers:
        plot_active_servers(axes[idx], power_df)
        idx += 1
    if include_jobs:
        plot_waiting_jobs(axes[idx], power_df)
        idx += 1
    if include_qos:
        plot_real_time_qos(axes[idx], power_df)

    # Common x-axis label for all plots
    fig.text(0.5, 0.02, 'Time (seconds)', ha='center', fontsize=16, fontweight='bold')

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.04, 1, 1])  # Make space for the shared x-axis label

    # Save as high-quality PDF
    plt.savefig(f'plots/{file_name_prefix}_power_consumption.pdf', format='pdf', dpi=300, bbox_inches='tight')

    # Show the combined plot
    plt.show()

def plot_collaborative_and_individual_traces(collab_power_trace, power_traces, job_tables=None,
                                             file_name_prefix="collab_and_individual",
                                             labels=None,
                                             legend_outside=False):
    # Style settings to match monetary_cost_dc_level.py
    plt.rcParams.update({
        # "font.family": "serif",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 20,
        "axes.grid": True,
        "grid.alpha": 0.8,
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white"
    })
    if job_tables:
        fig, axes = plt.subplots(4, 1, figsize=(22, 14), sharex=False,
                                 gridspec_kw={'height_ratios': [1.0, 1.0, 0.5, 0.8]})
    else:
        fig, axes = plt.subplots(3, 1, figsize=(22, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [1.0, 1.0, 0.5]})

    # Set white background and black spines
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        ax.set_axisbelow(True)

    # ========== 1. Collaborative Power tracking ==========
    axes[0].plot(collab_power_trace['time'], collab_power_trace['realSum']/1000, label=r"$P_\text{actual} \text{ (Collaborative)} $",
                 color='blue', linewidth=3)
    axes[0].plot(collab_power_trace['time'], collab_power_trace['target']/1000, '--', label=r"$P_\text{target} \text{ (Collaborative)} $",
                 color='red', linewidth=3)
    axes[0].set_ylabel('Power (kW)', fontsize=20)
    axes[0].legend(fontsize=20, loc="best", frameon=True, ncol=2)
    axes[0].grid(True, linestyle='--', alpha=0.8, color='gray', linewidth=1.0)

    # ========== 2. Individual Power traces ==========
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:cyan']
    line_styles = ['--', ':', '-' , '-.', '-', '--']
    for idx, power_trace in enumerate(power_traces):
        color = colors[idx % len(colors)]  # cycle colors if needed
        axes[1].plot(power_trace['time'], power_trace['realSum']/1000,
                     label=fr"$P_\text{{actual}}$ ({labels[idx]})", color=color, linewidth=3)
        axes[1].plot(power_trace['time'], power_trace['target']/1000,
                     linestyle=line_styles[idx], color='red',
                     label=fr"$P_\text{{target}}$ ({labels[idx]})", linewidth=2)
    axes[1].set_ylabel('Power (kW)', fontsize=20)
    axes[1].legend(fontsize=20, loc="upper right", frameon=True, ncol=4)
    axes[1].grid(True, linestyle='--', alpha=0.8, color='gray', linewidth=1.0)

    # ========== 3. Real-Time QoS ==========
    for idx, power_trace in enumerate(power_traces):
        color = colors[idx % len(colors)]
        axes[2].plot(power_trace['time'], power_trace['realtimeQoS'],
                     label=f'{labels[idx]}', color=color, linewidth=3)
    axes[2].set_ylabel('Real-Time QoS', fontsize=18)
    axes[2].legend(fontsize=20, loc="lower right", frameon=True, ncol=2)
    axes[2].grid(True, linestyle='--', alpha=0.8, color='gray', linewidth=1.0)

    # ========== 4. Job Arrivals (if job_tables provided) ==========
    if job_tables is not None:
        def count_arrivals_per_5min(arrival_times):
            minutes = arrival_times.astype(int) // 60
            bins = minutes // 5
            return bins.value_counts().reindex(range(0, 12), fill_value=0).sort_index()
        bar_width = 0.35
        x_positions = np.arange(12)
        for idx, job_table in enumerate(job_tables):
            color = colors[idx % len(colors)]
            arrival_times = job_table['arrival_time']
            arrival_times = arrival_times[arrival_times > 0]
            counts = count_arrivals_per_5min(arrival_times)
            position = x_positions + (idx - len(job_tables) / 2 + 0.5) * bar_width
            bars = axes[3].bar(position, counts,
                               width=bar_width, alpha=0.7,
                               label=f'{labels[idx]}',
                               color=color)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    axes[3].text(bar.get_x() + bar.get_width() / 2., height,
                                 f'{int(height):,}',
                                 ha='center', va='bottom', fontsize=10)
        axes[3].set_ylabel('Jobs per 5min', fontsize=16)
        axes[3].set_xlabel('Time (minutes)', fontsize=16)
        axes[3].set_xticks(x_positions)
        axes[3].set_xticklabels([f'{i * 5}-{(i + 1) * 5}' for i in range(12)])
        axes[3].legend(fontsize=14, loc="best", frameon=True)
        axes[3].grid(True, linestyle='--', alpha=0.8, color='gray', linewidth=1.0)

    axes[-1].set_xlabel('Time (seconds)', fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 1])
    print("Saving plot in" + f"plots/{file_name_prefix}.pdf")
    plt.savefig(f"plots/{file_name_prefix}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
