import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "notebooks"))
from plotting_helpers import get_power_trace_df, get_job_table_df


DEFAULT_OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..", "output", "simulation")


def plot(output_dir, save_dir="plots"):
    power_df = get_power_trace_df(output_dir)
    exp_name = os.path.basename(os.path.normpath(output_dir))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(22, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.6, 0.6]},
    )

    ax1.plot(power_df["time"], power_df["realSum_kw"], label="Power Consumption", linewidth=2.5, color="blue")
    ax1.plot(power_df["time"], power_df["target_kw"], label="Power Target", color="red", linewidth=2, linestyle="--")
    ax1.set_ylabel("Power (kW)", fontsize=16)
    ax1.legend(fontsize=14, loc="upper right")
    ax1.grid(True, linestyle="--", linewidth=0.6)
    ax1.tick_params(axis="both", which="major", labelsize=14)

    ax2.plot(power_df["time"], power_df["active_servers"], label="Active Servers", color="green", linewidth=2)
    ax2.set_ylabel("Active Servers", fontsize=16)
    ax2.legend(fontsize=14, loc="upper left")
    ax2.grid(True, linestyle="--", linewidth=0.6)
    ax2.tick_params(axis="both", which="major", labelsize=14)

    ax3.plot(power_df["time"], power_df["waitingSum"], label="Jobs in Queue", color="purple", linewidth=2)
    ax3.set_ylabel("Waiting Jobs", fontsize=16)
    ax3.legend(fontsize=14, loc="upper right")
    ax3.grid(True, linestyle="--", linewidth=0.6)
    ax3.tick_params(axis="both", which="major", labelsize=14)

    fig.text(0.5, 0.02, "Time (seconds)", ha="center", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{exp_name}_power_consumption.pdf")
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot simulation output")
    parser.add_argument("--simulation-output", type=str, required=True, help="Name of the simulation output directory")
    default_save_dir = os.path.join(os.path.dirname(__file__), "plots")
    parser.add_argument("--save-dir", default=default_save_dir, help="Directory to save the plot PDF")
    args = parser.parse_args()

    output_dir = os.path.join(DEFAULT_OUTPUT_BASE, args.simulation_output)
    plot(output_dir, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
