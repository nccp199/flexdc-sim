import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set ACM paper style parameters
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 11,
    'lines.linewidth': 1.5,
    'grid.linewidth': 0.6,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none'
})

# Constants and inputs
iso_data_dir = "../../../data/"
start_hour = 16  # Start hour (customizable)
end_hour = 17    # End hour (customizable)
data_points_per_hour = 1800  # Assumption: 1800 data points (2-second intervals) per hour
start_idx = start_hour * data_points_per_hour  # Start index for given time range
end_idx = end_hour * data_points_per_hour  # End index for given time range

# Load dynamic and traditional signal datasets
dyn = pd.read_csv(iso_data_dir + "1-8-2025_dynamic.csv", header=None)[0]
trd = pd.read_csv(iso_data_dir + "1-8-2025_traditional.csv", header=None)[0]

# Filter data for the specified time range
dyn_filtered = dyn[start_idx:end_idx]
trd_filtered = trd[start_idx:end_idx]

# Create numerical x-axis
x_axis = range(len(dyn_filtered))

# Generate time labels for display
time_labels = [f"{start_hour + i//data_points_per_hour:02d}:{(i % data_points_per_hour) // 30:02d}"
               for i in range(len(dyn_filtered))]

# ACM single-column width figure (slightly wider than IEEE)
fig, ax = plt.subplots(figsize=(6.0, 3), facecolor='white')
ax.set_facecolor('white')

# ACM-style colors (energy-themed, colorblind-friendly)
dynamic_color = 'darkgreen'    # Professional blue
traditional_color = 'black'  # Contrasting magenta
grid_color = '#CCCCCC'       # Light gray

# Plot with ACM-appropriate colors and styles
ax.plot(x_axis, dyn_filtered, label='Dynamic (RegD)', color=dynamic_color,
        linestyle='--', linewidth=2.0, alpha=0.9)
ax.plot(x_axis, trd_filtered, label='Traditional (RegA)', color=traditional_color,
        linestyle='-', linewidth=2.0, alpha=0.9)

num_ticks = 7  # 16:00, 16:10, 16:20, 16:30, 16:40, 16:50, 17:00
tick_positions = []
tick_labels = []

for i in range(num_ticks):
    minutes = i * 10
    hour = start_hour
    minute = minutes

    # Handle hour rollover
    if minute >= 60:
        hour = start_hour + 1
        minute = minute - 60

    # Calculate position in the data
    if hour == start_hour:
        pos = int((minute / 60) * len(time_labels))
    else:
        pos = len(time_labels) - 1  # End of data

    tick_positions.append(pos)
    tick_labels.append(f"{hour}:{minute:02d}")

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=0)
ax.set_xlabel('Time', fontweight='normal')
ax.set_ylabel('Regulation Signal y(t)')

# Professional grid
ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.6, color=grid_color)
ax.set_axisbelow(True)

# Place legend outside at the top, centered, with two columns
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.18),
    ncol=2,
    frameon=True,
    fancybox=False,
    shadow=False,
    framealpha=0.95,
    edgecolor='#666666',
    facecolor='white',
    columnspacing=1.0
)

# Frame from all sides
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')

# Set axis limits for better visualization
ax.set_xlim(0, len(x_axis)-1)
y_min, y_max = min(min(dyn_filtered), min(trd_filtered)), max(max(dyn_filtered), max(trd_filtered))
y_margin = (y_max - y_min) * 0.1
ax.set_ylim(y_min - y_margin, y_max + y_margin)

# Professional layout
plt.tight_layout(pad=0.2)

# Save in ACM-preferred formats
# plt.savefig("acm_energy_signal_comparison.pdf", format="pdf", dpi=300, bbox_inches="tight",
#             pad_inches=0.02, facecolor='white', edgecolor='none')
plt.savefig("plots/iso_signal_comparison.png", format="png", dpi=300, bbox_inches="tight",
            pad_inches=0.02, facecolor='white', edgecolor='none')

plt.show()
