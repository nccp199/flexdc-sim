
def calculate_energy_cost(experiment_summary,experiment_duration_hour, baseline_power=0):
    P = experiment_summary['P']
    dr_program = experiment_summary['DR Program']

    hour = 3600
    kwh = hour * 1000
    piP, piR, piE, piI = 0.1 / kwh, 0.1 / kwh, 0.1 / kwh, 0.07 / kwh
    total_cost = 0

    if dr_program == 'RSR':
        R = experiment_summary['R']
        average_tracking_error = experiment_summary['Average Tracking Error']
        power_cost = (piP * P - piR * R) * hour * experiment_duration_hour
        tracking_cost = (piE * R * average_tracking_error) * hour * experiment_duration_hour
        total_cost = power_cost + tracking_cost
        print(total_cost)
    elif dr_program == 'EDR':
        P_max = baseline_power
        print("No DR baseline ", P_max)
        print("P:", P)
        total_cost = (piP * P - piI * (P_max - P)) * hour * experiment_duration_hour
        print(total_cost)
    elif dr_program == 'NO DR':
        print("NO DR")
        total_cost = (piP * P) * hour * experiment_duration_hour
        print(total_cost)

    return total_cost

def calculate_normalized_energy_costs(metrics_df):
    grouped = metrics_df.groupby("Workload")

    # Adjust NoDR cost and store in Normalized Cost
    def adjust_nodr_cost(group):
        nodr_row = group[group["DR Program"] == "NO DR"]
        if not nodr_row.empty:
            nodr_energy = nodr_row["Energy"].values[0]
            nodr_cost = nodr_row["Cost"].values[0]
            for index, row in group.iterrows():
                if row["DR Program"] != "NO DR":
                    energy_diff_percentage = (nodr_energy - row["Energy"]) / nodr_energy
                    adjusted_cost = nodr_cost * (1 - energy_diff_percentage)
                    group.loc[group["DR Program"] == "NO DR", "Normalized Cost"] = adjusted_cost
                else:
                    group.loc[index, "Normalized Cost"] = row["Cost"]
        return group

    adjusted_data = grouped.apply(adjust_nodr_cost)
    return adjusted_data