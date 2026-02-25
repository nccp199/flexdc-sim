import os
import pandas as pd


def get_tracking_error_90th(power_trace_df):
    """
    Get tracking error value at 90th percentile using power trace.
    """
    error_at_90 = power_trace_df['trackingError'].abs().quantile(0.9)
    return error_at_90


def calculate_tracking_error_90th(power_trace_df, total_reserve):
    """
    Calculate tracking error based on power trace.
    """
    return abs(power_trace_df['realSum'] - power_trace_df['target']).quantile(0.9) / total_reserve


def calculate_tracking_error_distribution(power_trace_df, total_reserve):
    return abs(power_trace_df['realSum'] - power_trace_df['target']) / total_reserve


def get_write_tracking_error_90th(power_trace_df, output_dir):
    """
    Write tracking error to CSV file.
    """
    tracking_error_90th = get_tracking_error_90th(power_trace_df)
    output_file = os.path.join(output_dir, 'tracking_error_90th.csv')

    # Save the value to a CSV file
    tracking_error_90th_df = pd.DataFrame({'Tracking Error 90th Percentile': [tracking_error_90th]})
    tracking_error_90th_df.to_csv(output_file, index=False)

    return tracking_error_90th
