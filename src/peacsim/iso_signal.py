import os.path

import pandas as pd
import numpy as np


def read_iso_signal(experiment_config, simulation_duration, iso_signal_granularity, normalize=False):
    # Returns a slice of iso signal from a random offset to the offset + (simulation_duration // iso_signal_granularity)
    hour_seconds = 3600
    filepath = experiment_config.iso_file_path
    iso_signal = pd.read_csv(filepath, header=None, dtype=float)
    iso_signal = iso_signal[0].tolist()
    if normalize:
        iso_signal /= np.max(np.abs(iso_signal[:simulation_duration//iso_signal_granularity]))
        ## TODO: this is done to mimic the old simulator. may be restored

    num_samples_required = (simulation_duration // iso_signal_granularity) + 1
    if experiment_config.randomize_iso_start:
        max_offset = len(iso_signal) - num_samples_required
        if max_offset < 0:
            raise ValueError("The ISO signal is shorter than the simulation duration.")
        offset = np.random.randint(0, max_offset + 1)
        iso_signal = iso_signal[offset: offset + num_samples_required]
        print("Offset into ISO signal:", offset)
    else:
        start_offset_hour = experiment_config.iso_signal_start_hour if experiment_config.iso_signal_start_hour else 0
        start_index = int(start_offset_hour * (hour_seconds/iso_signal_granularity))
        iso_signal = iso_signal[start_index:start_index+num_samples_required]
    return iso_signal


def calculate_cluster_power_caps(iso_signal, P_bar, R, phase, simulation_duration, iso_signal_granularity):
    start = (phase % 24) * simulation_duration // iso_signal_granularity
    end = (phase % 24 + 1) * simulation_duration // iso_signal_granularity

    iso_signal_segment = np.array(iso_signal[start:end + 1])
    caps = P_bar + iso_signal_segment * R
    return caps.tolist()
