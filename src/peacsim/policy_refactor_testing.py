#!/usr/bin/env python3
"""
policy_refactor_testing.py
Compares outputs between old AQA_runtime_policy.py and new refactored policies.

Usage:
    python policy_refactor_testing.py [policy_name]

Examples:
    python policy_refactor_testing.py              # Test all policies
    python policy_refactor_testing.py AQA          # Test only AQA
    python policy_refactor_testing.py flex-resource # Test only flex-resource

Run from: src/peacsim/
"""

import subprocess
import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# ============ TEST CONFIGURATIONS ============
TEST_CONFIGS = {
    'AQA': {
        'experiment_config': '../../configs/experiment/new_iso/dynamic_signal/exp_inf_low_util=0.6.ini',
        'cluster_config': '../../configs/cluster/cluster.ini',
        'policy_config': 'output/optimization/opt_SA_inf_low_util_RSR_1000server_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_dynamic_signal_20250816201226/optimal_policy.ini',
        'job_config': '../../configs/workload/W2-short-qos5_4.5_4_3.5.ini',
        'qos_percentile': '0.9',
    },
    'priority-cap': {
        'experiment_config': '../../configs/experiment/new_iso/traditional_signal/exp_inf_low_util=0.6.ini',
        'cluster_config': '../../configs/cluster/cluster.ini',
        'policy_config': 'output/optimization/opt_pcap=0.5_node_state=false_SA_inf_low_util_EDR_1000server_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_traditional_signal_20250912094602/optimal_policy.ini',
        'job_config': '../../configs/workload/W2-short-qos5_4.5_4_3.5.ini',
        'qos_percentile': '0.9',
    },
    'proportional-cap': {
        'experiment_config': '../../configs/experiment/new_iso/traditional_signal/exp_inf_low_util=0.6.ini',
        'cluster_config': '../../configs/cluster/cluster.ini',
        'policy_config': 'output/optimization/opt_proportional-cap_node_state=false_SA_inf_low_util_EDR_1000server_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_traditional_signal_20250916083802/optimal_policy.ini',
        'job_config': '../../configs/workload/W2-short-qos5_4.5_4_3.5.ini',
        'qos_percentile': '0.9',
    },
    'flex-resource': {
        'experiment_config': '../../configs/experiment/new_iso/dynamic_signal/exp_inf_low_util=0.6.ini',
        'cluster_config': '../../configs/cluster/cluster.ini',
        'policy_config': 'output/optimization/opt_flex-resource_probabilities=0.8,0.8,0.8,0.8_SA_inf_low_util_RSR_1000server_exp_inf_low_util=0.6_W2-short-qos5_4.5_4_3.5_dynamic_signal_20250915100931/optimal_policy.ini',
        'job_config': '../../configs/workload/W2-short-qos5_4.5_4_3.5.ini',
        'qos_percentile': '0.9',
    },
}


# =============================================


def run_simulation(script_name: str, output_prefix: str, timestamp: str, config: dict) -> str:
    """Run a simulation and return the output directory path."""
    output_name = f"{output_prefix}_{timestamp}"

    cmd = [
        sys.executable, '-u', script_name,
        '--experiment-config', config['experiment_config'],
        '--cluster-config', config['cluster_config'],
        '--policy-config', config['policy_config'],
        '--job-config', config['job_config'],
        '--qos-percentile', config['qos_percentile'],
        '--output-dir', output_name,
    ]

    print(f"Running: {script_name}")
    print(f"  Output: {output_name}")

    env = os.environ.copy()
    env['PYTHONPATH'] = '..'

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR running {script_name}:")
        print(result.stderr)
        return None
    print(result.stdout.strip())

    pattern = f"output/simulation/{output_name}_*"
    dirs = sorted(glob.glob(pattern), reverse=True)

    if not dirs:
        print(f"ERROR: Could not find output directory matching {pattern}")
        return None

    return dirs[0]


def compare_csv_files(old_file: str, new_file: str, file_name: str, tolerance: float = 1e-6) -> dict:
    """Compare two CSV files and return comparison results."""
    results = {
        'file': file_name,
        'identical': False,
        'exists_old': os.path.exists(old_file),
        'exists_new': os.path.exists(new_file),
        'differences': [],
        'max_diff': 0.0,
        'max_diff_column': None,
    }

    if not results['exists_old'] or not results['exists_new']:
        return results

    try:
        old_df = pd.read_csv(old_file)
        new_df = pd.read_csv(new_file)
    except Exception as e:
        results['differences'].append(f"Error reading files: {e}")
        return results

    if old_df.shape != new_df.shape:
        results['differences'].append(
            f"Shape mismatch: OLD={old_df.shape}, NEW={new_df.shape}"
        )
        return results

    if list(old_df.columns) != list(new_df.columns):
        results['differences'].append("Column names differ")
        return results

    numeric_cols = old_df.select_dtypes(include=[np.number]).columns
    all_match = True

    for col in numeric_cols:
        diff = np.abs(old_df[col] - new_df[col])
        max_col_diff = diff.max()

        if max_col_diff > results['max_diff']:
            results['max_diff'] = max_col_diff
            results['max_diff_column'] = col

        if max_col_diff > tolerance:
            all_match = False
            diff_mask = diff > tolerance
            diff_indices = old_df[diff_mask].index[:3]

            for idx in diff_indices:
                results['differences'].append(
                    f"  {col}[{idx}]: OLD={old_df.loc[idx, col]:.6f}, NEW={new_df.loc[idx, col]:.6f}"
                )

    results['identical'] = all_match
    return results


def print_comparison_results(results: dict):
    """Pretty print comparison results."""
    file_name = results['file']

    if not results['exists_old']:
        print(f"  ⚠ {file_name}: Not found in OLD output")
        return
    if not results['exists_new']:
        print(f"  ⚠ {file_name}: Not found in NEW output")
        return

    if results['identical']:
        print(f"  ✓ {file_name}: IDENTICAL (max diff: {results['max_diff']:.2e})")
    else:
        print(f"  ✗ {file_name}: DIFFERENT")
        if results['max_diff_column']:
            print(f"    Max difference: {results['max_diff']:.6f} in column '{results['max_diff_column']}'")
        for diff in results['differences'][:10]:
            print(f"    {diff}")


def run_policy_test(policy_name: str, config: dict) -> bool:
    """Test a single policy and return True if outputs match."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    print("\n" + "=" * 60)
    print(f"Testing Policy: {policy_name}")
    print("=" * 60)

    # Check policy config exists
    if not os.path.exists(config['policy_config']):
        print(f"  ⚠ SKIPPED: Policy config not found: {config['policy_config']}")
        return None

    # Run OLD simulation
    print(f"\nRunning OLD code for {policy_name}...")
    old_dir = run_simulation('run_simulator_old.py', f'OLD_{policy_name}', timestamp, config)

    if not old_dir:
        print(f"Failed to run OLD simulation for {policy_name}")
        return False

    # Run NEW simulation
    print(f"\nRunning NEW code for {policy_name}...")
    new_dir = run_simulation('run_simulator.py', f'NEW_{policy_name}', timestamp, config)

    if not new_dir:
        print(f"Failed to run NEW simulation for {policy_name}")
        return False

    # Compare outputs
    print(f"\nComparing outputs for {policy_name}:")
    print(f"  OLD: {old_dir}")
    print(f"  NEW: {new_dir}")

    files_to_compare = ['power_trace.csv', 'job_table.csv']
    all_identical = True

    for file_name in files_to_compare:
        old_file = os.path.join(old_dir, file_name)
        new_file = os.path.join(new_dir, file_name)
        results = compare_csv_files(old_file, new_file, file_name)
        print_comparison_results(results)
        if not results['identical']:
            all_identical = False

    return all_identical


def main():
    print("=" * 60)
    print("Policy Refactoring Test Suite")
    print("=" * 60)

    # Check scripts exist
    if not os.path.exists('run_simulator_old.py'):
        print("ERROR: run_simulator_old.py not found!")
        sys.exit(1)
    if not os.path.exists('run_simulator.py'):
        print("ERROR: run_simulator.py not found!")
        sys.exit(1)

    # Determine which policies to test
    if len(sys.argv) > 1:
        policies_to_test = [sys.argv[1]]
        if policies_to_test[0] not in TEST_CONFIGS:
            print(f"ERROR: Unknown policy '{policies_to_test[0]}'")
            print(f"Available policies: {list(TEST_CONFIGS.keys())}")
            sys.exit(1)
    else:
        policies_to_test = list(TEST_CONFIGS.keys())

    # Run tests
    results = {}
    for policy_name in policies_to_test:
        config = TEST_CONFIGS[policy_name]
        results[policy_name] = run_policy_test(policy_name, config)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for policy_name, passed in results.items():
        if passed is None:
            status = "⚠ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {policy_name}: {status}")

    # Exit code
    failed = [p for p, r in results.items() if r is False]
    if failed:
        print(f"\n{len(failed)} policy test(s) failed: {failed}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())