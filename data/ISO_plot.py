import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
csv_filename = 'ISO24h.csv'  # Path to your CSV file
data_point_interval = 4  # seconds per data point
num_plots = 6  # We'll split the data into 6 segments

# --- Read CSV Data ---
# Assumes the CSV file contains one value per line without a header
data = pd.read_csv(csv_filename, header=None).iloc[:, 0].values

# --- Split Data into 6 Segments ---
# np.array_split splits the array into nearly equal parts even if it doesn't divide evenly.
segments = np.array_split(data, num_plots)

# --- Create a 2x3 Grid of Subplots ---
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axes = axes.flatten()  # Flatten the 2D array for easy iteration

# --- Plot Each Segment ---
for i, segment in enumerate(segments):
    # Generate a time axis for the segment.
    # Here we use a relative time axis (starting at 0 for each segment).
    time_axis = np.arange(len(segment)) * data_point_interval

    # Optionally, to display the global time (offset by the segment's position),
    # you can uncomment the following two lines and comment out the relative time_axis line above.
    # start_idx = i * len(segment)
    # time_axis = (np.arange(len(segment)) + start_idx) * data_point_interval

    axes[i].plot(time_axis, segment, lw=1)

    # Compute start and end times for display (in hours)
    start_time = 0 if i == 0 else (i * len(segment) * data_point_interval) / 3600
    end_time = ((i * len(segment) + len(segment) - 1) * data_point_interval) / 3600
    axes[i].set_title(f'Segment {i + 1}: {start_time:.2f} to {end_time:.2f} hours')
    axes[i].set_xlabel('Time (seconds)')
    axes[i].set_ylabel('Signal Value')
    axes[i].grid(True)

plt.tight_layout()
plt.show()
