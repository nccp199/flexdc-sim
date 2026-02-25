import pandas as pd
from src.peacsim.iso_signal import read_iso_signal, calculate_cluster_power_caps
from src.peacsim.parsing import experiment_config_reader as experiment_parser
from src.peacsim.notebooks.plotting_helpers import *
from src.peacsim.calculate_monetary_cost import *
from src.peacsim.extract_power_metrics import *
from src.peacsim.energy_cost_calculator import calculate_energy_cost

def read_experiment_data(result_sheet, return_power_traces=False, return_job_tables=False):
    """
    Reads experiment data from the specified directories and returns a DataFrame with metrics.

    Args:
        simulation_result_dirs (list): List of directories containing simulation results.
        simulation_names (list): List of names corresponding to each simulation result directory.

    Returns:
        pd.DataFrame: DataFrame containing the metrics for each simulation.
    """
    result_sheet = result_sheet[result_sheet['Sim Output Dir'].notna()]
    simulation_names = result_sheet["Sim Output Dir"].tolist()

    output_path = "../output/simulation/"
    simulation_result_dirs = [output_path + dir + "/" for dir in simulation_names]

    power_traces = []
    job_tables = []

    metrics_df = pd.DataFrame()
    for index in range(len(result_sheet)):
        current_experiment = result_sheet.iloc[index]
        experiment_config = experiment_parser.ExperimentConfigReader(simulation_result_dirs[0]
                                                                     + "/experiment_config.ini")

        simulation_result_dir = output_path + current_experiment['Sim Output Dir'] + "/"

        power_trace = get_power_trace_df(simulation_result_dir)
        job_table = get_job_table_df(simulation_result_dir)
        qos_summary = pd.read_csv(f"{simulation_result_dir}qos_summary.csv")

        # cost = pd.read_csv(f"{simulation_result_dir}cost_power_rsr.csv", header=None).iloc[0, 0]



        # Read pr_df
        pr_df = pd.read_csv(f"{simulation_result_dir}PRtable.csv")

        # Read tracking error
        tracking_error_at_90th = pd.read_csv(f"{simulation_result_dir}tracking_error_90th.csv").iloc[0, 0]
        average_tracking_error_dc = calculate_tracking_error_distribution(power_trace, pr_df.iloc[0]['R']).mean()
        baseline_power = -1
        p_max_no_dr = -1
        if current_experiment['DR Program'] == 'NO DR':
            pr_df.loc[0, 'P'] = power_trace['realSum'].mean()
            pr_df.loc[0, 'R'] = -1
            p_max_no_dr = power_trace['realSum'].max()
            average_tracking_error_dc = -1
            tracking_error_at_90th = -1

        elif current_experiment['DR Program'] == 'EDR':
            baseline_power = result_sheet[
                (result_sheet['Workload'] == current_experiment["Workload"]) &
                (result_sheet['DR Program'] == "NO DR") &
                (result_sheet['QoS Percentile'] == current_experiment["QoS Percentile"])
            ]['P_max_NoDR'].values[0]
        elif current_experiment['DR Program'] == 'RSR':
            pass

        qos_violation_ratio = (qos_summary['qos_' + str(current_experiment['QoS Percentile']) + 'th'] > qos_summary['qos_threshold']).sum() / len(qos_summary) * 100
        # Combine cost, tracking error, and pr_df into a single row

        combined_row = pd.DataFrame({
            "Workload": current_experiment['Workload'],
            "Simulation": simulation_names[index],
            "DR Program": current_experiment['DR Program'],
            "P_max_no_dr": p_max_no_dr,
            "P": pr_df.iloc[0]['P'],
            "R": pr_df.iloc[0]['R'],
            "Average Tracking Error": [average_tracking_error_dc],
            "90th Tracking Error": [tracking_error_at_90th],
            "QoS Violation Ratio": [qos_violation_ratio],
            "Energy": [power_trace['realSum'].sum()],
        })

        monetary_cost = calculate_energy_cost(combined_row.iloc[0],
                                              experiment_config.simulation_duration / 3600,
                                              baseline_power=baseline_power)
        combined_row["Cost"] = monetary_cost

        metrics_df = pd.concat([metrics_df, combined_row], ignore_index=True)

        power_traces.append(power_trace)
        job_tables.append(job_table)



    return_dict = {"metrics_df": metrics_df}

    if return_power_traces:
        return_dict['power_traces'] = power_traces
    if return_job_tables:
        return_dict['job_tables'] = job_tables

    return return_dict
