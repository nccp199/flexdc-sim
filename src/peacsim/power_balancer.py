import pandas as pd
import numpy as np
from iso_signal import read_iso_signal, calculate_cluster_power_caps


class PowerBalancer:
    def __init__(self, num_data_centers,
                 policy_configs,
                 experiment_configs,
                 job_configs,
                 stop_signal,
                 assignment_conditions=None, power_balancer_condition=None, qos_write_lock=None):

        self.num_data_centers = num_data_centers

        self.power_bids = [policy_configs[i].P for i in range(num_data_centers)]
        self.reserve_bids = [policy_configs[i].R for i in range(num_data_centers)]

        iso_signal = read_iso_signal(experiment_configs[0],
                                     experiment_configs[0].simulation_duration,
                                     experiment_configs[0].iso_signal_granularity,
                                     experiment_configs[0].normalize_iso_signal)
        self.iso_signal_granularity = experiment_configs[0].iso_signal_granularity
        self.collaborative_power_target = calculate_cluster_power_caps(iso_signal,
                                                                  sum(self.power_bids),
                                                                  sum(self.reserve_bids),
                                                                  0,
                                                                  experiment_configs[0].simulation_duration,
                                                                  experiment_configs[0].iso_signal_granularity)

        self.ratio_power_bids = np.asarray(self.power_bids) / sum(self.power_bids)
        self.stop_signal = stop_signal
        self.assignment_events = assignment_conditions
        self.power_balancer_condition = power_balancer_condition
        self.qos_write_lock = qos_write_lock
        self.tau = 0.5
        self.phi = 0.5

        dc_weight_dfs = [pd.read_csv(policy_configs[i].policy_parameters['weights_path'])
                                     for i in range(num_data_centers)]
        dc_weights = [{index: float(row[1]) for index, row in df.iterrows()} for df in dc_weight_dfs]
        self.estimated_power_capacities = []
        for dc_index in range(num_data_centers):
            self.estimated_power_capacities.append(
                experiment_configs[dc_index].server_count * sum(dc_weights[dc_index][job_type] * job_configs[dc_index].all_max_job_power[job_type]
                                                               for job_type in dc_weights[dc_index])
            )
        self.estimated_power_capacities = np.asarray(self.estimated_power_capacities)
        self.all_dc_capacity = float(sum(self.estimated_power_capacities))

    def cap_and_redistribute(self, initial_assignments):
        """
        Vectorised version – caps per-DC targets, redistributes any overflow
        in a single pass, drops un-allocatable residue.
        Returns a NumPy array (caller can .tolist() if needed).
        """

        # 1.  First pass – cap at individual capacities
        assigments = initial_assignments.copy()
        assignments = np.asarray(assigments, dtype=float)
        caps = self.estimated_power_capacities  # ndarray

        excess = np.clip(assignments - caps, 0.0, None)  # >0 where over capacity
        overflow = float(excess.sum())  # scalar
        assignments -= excess  # now within caps

        # 2.  Single vectorised redistribution step (if any overflow left)
        if overflow > 1e-6:
            headroom = np.clip(caps - assignments, 0.0, None)  # free capacity
            total_headroom = float(headroom.sum())

            if total_headroom > 0.0:
                # If overflow exceeds head-room we only fill the head-room;
                # any residue is discarded automatically.
                factor = min(1.0, overflow / total_headroom)  # 0 ≤ factor ≤ 1
                assignments += factor * headroom  # proportional fill

        return assignments.tolist()

    def run(self, current_time_shared, qos_values, power_assignments):
        print("[Power Balancer] Starting power balancer loop...")
        while not self.stop_signal.value:

            with self.power_balancer_condition:
                self.power_balancer_condition.wait(timeout=0.1)
            # print(f"[PowerBalancer] Received QoS values. Current keys: {list(qos_values.keys())}")
            if self.stop_signal.value:
                break

            # Proceed only when all DCs have posted
            if len(qos_values) < self.num_data_centers:
                continue  # Wait for more, next cycle will notify again
            # Copy current QoS snapshot
            qos_snapshot = dict(qos_values)

            # Example strategy: more QoS (i.e., more urgency) gets more power
            total_qos = sum(qos_snapshot.values()) + 1e-6  # avoid divide-by-zero

            total_power_budget = self.collaborative_power_target[current_time_shared.value // self.iso_signal_granularity]
            effective_power_budget = total_power_budget
            # effective_power_budget = min(total_power_budget, self.all_dc_capacity)

            raw_assignments = [
                effective_power_budget * (
                        self.tau * (qos_snapshot[dc] / total_qos) +
                        self.phi * self.ratio_power_bids[dc]
                )
                for dc in range(self.num_data_centers)
            ]

            # feasible_assignments = self.cap_and_redistribute(raw_assignments)

            for dc_id, target in enumerate(raw_assignments):
                power_assignments[dc_id] = target

            # Optional: clear qos_values if you want to force re-population
            for dc_id in range(self.num_data_centers):
                with self.qos_write_lock:
                    del qos_values[dc_id]

            for dc_id in range(self.num_data_centers):
                # with self.assignment_conditions[dc_id]:
                    # self.assignment_conditions[dc_id].notify()
                self.assignment_events[dc_id].set()

            # print(f"[PowerBalancer] Starts waiting: {list(qos_values.keys())}")

            # time.sleep(0.01)  # small pause to prevent tight loop
        print("[Power Balancer] Power balancer terminated...")