import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_gradient_descent(csv_path, dc_index=""):

    # Read the CSV file
    data = pd.read_csv(csv_path)

    # Extract columns
    iter = data['iter']
    P = data['P']
    R = data['R']
    Pscale = data['P'] / data['Pcoef']
    Rscale = data['R'] / data['Rcoef']
    CostTheory = data['cost_theory']
    costPower = data['costPower']
    costError = data['cost_error']
    costQoS = data['cost_QoS']

    # Create a figure with 8 subplots in a 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))

    # Plot P against iter (top-left)
    axes[0, 0].plot(iter, P, '-')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('P')
    axes[0, 0].set_title('P vs. Iteration')
    axes[0, 0].grid(True)

    # Plot R against iter (second from top-left)
    axes[1, 0].plot(iter, R, '-')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('R')
    axes[1, 0].set_title('R vs. Iteration')
    axes[1, 0].grid(True)

    # Plot CostTheory against iter (third from top-left)
    axes[2, 0].plot(iter, CostTheory, '-')
    axes[2, 0].set_xlabel('Iteration')
    axes[2, 0].set_ylabel('CostTheory')
    axes[2, 0].set_title('CostTheory vs. Iteration')
    axes[2, 0].grid(True)

    # Plot costPower against iter (bottom-left)
    axes[3, 0].plot(iter, costPower, '-')
    axes[3, 0].set_xlabel('Iteration')
    axes[3, 0].set_ylabel('CostPower')
    axes[3, 0].set_title('CostPower vs. Iteration')
    axes[3, 0].grid(True)

    # Plot Pscale against iter (top-right)
    axes[0, 1].plot(iter, Pscale, '-')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Pscale')
    axes[0, 1].set_title('Pscale vs. Iteration')
    axes[0, 1].grid(True)

    # Plot Rscale against iter (second from top-right)
    axes[1, 1].plot(iter, Rscale, '-')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Rscale')
    axes[1, 1].set_title('Rscale vs. Iteration')
    axes[1, 1].grid(True)

    # Plot costQoS against iter (bottom-right)
    axes[2, 1].plot(iter, costQoS, '-')
    axes[2, 1].set_xlabel('Iteration')
    axes[2, 1].set_ylabel('CostQoS')
    axes[2, 1].set_title('CostQoS vs. Iteration')
    axes[2, 1].grid(True)

    # Plot costError against iter (third from top-right)
    axes[3, 1].plot(iter, costError, '-')
    axes[3, 1].set_xlabel('Iteration')
    axes[3, 1].set_ylabel('CostError')
    axes[3, 1].set_title('CostError vs. Iteration')
    axes[3, 1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot to the same directory as the CSV file
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, 'GD_plots' + dc_index + '.png')
    plt.savefig(output_path)

    # Show the plots
    plt.show()

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    experiment_name = 'W10_optim_test_20250204214410'
    plot_gradient_descent('output/optimization/' + experiment_name + '/GD.csv')

