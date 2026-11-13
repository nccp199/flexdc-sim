"""Collect compact native intervals and publish the S2a evaluation at finalize.

Three doubles are retained per elapsed step, plus budget-target samples for
optional external control. The JSON artifact holds the full-precision inputs
needed for replay; the cached public report is small.
Legacy CSV statistics and monetary costs are deliberately separate.
"""
from array import array
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from peacsim.metrics.tracking import (
    METRIC_VERSION, PowerInterval, TargetSchedule, evaluate_tracking,
    interval_from_transition,
)
from peacsim.metrics.energy_cost import evaluate_it_energy_cost


REPORT_VERSION = 'flexdc-tracking-report-v1'
LEGACY_VERSION = 'flexdc-legacy-csv-v0'
PAPER_SOURCE = 'https://doi.org/10.1145/3744255.3798112'


def paper_reference(result):
    """Paper Eq. (3): duration(error/R > 0.3) / duration < 0.1.

This is an explicitly named research reference, not a universal grid rule or
an SA objective. A P90 alone cannot decide the strict 10% boundary.
"""
    reference = dict(source=PAPER_SOURCE, equation=3, normalized_error_threshold=0.3,
                     maximum_exceedance_fraction=0.1, comparison='strictly_less_than',
                     status='not_applicable', exceedance_duration_seconds=None,
                     exceedance_fraction=None, passed=None)
    if result['normalization_status'] == 'applicable':
        bad = sum(s['duration_seconds'] for s in result['segments']
                  if abs(s['normalized_error']) > 0.3)
        duration = result['covered_duration_seconds']
        reference.update(status='applicable', exceedance_duration_seconds=bad,
                         exceedance_fraction=bad / duration, passed=10 * bad < duration)
    return reference


def write_report(path, payload):
    """Publish one complete JSON file; a failed write can be retried."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.tracking-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TrackingRecorder:
    def __init__(self, target_power_w, reserve_w, baseline_power_w, iso_step, duration, dt):
        # uuid4 uses OS entropy, never the simulator's Python/NumPy RNG streams.
        self.run_id = uuid4().hex
        self.duration, self.dt = duration, dt
        self.baseline_power_w = float(baseline_power_w)
        self.targets = (TargetSchedule(self.run_id, target_power_w, iso_step, 0, duration, reserve_w)
                        if duration > 0 else None)
        self.rows = array('d')
        self.interval_targets = array('d')
        self.target_step = math.gcd(int(iso_step), int(dt))
        self.has_external_budget = False
        self.energy_cost_report = None

    def append(self, transition, episode_id, *, external_it_budget_w=None):
        interval = interval_from_transition(transition, run_id=self.run_id,
                                            expected_episode_id=episode_id)
        if interval is not None:
            self.rows.extend((interval.start_seconds, interval.end_seconds, interval.actual_it_power_w))
            # A native fallback may change inside a physical interval. Keep
            # the common grid so the existing S2 evaluator can split it exactly.
            for time in range(interval.start_seconds, interval.end_seconds, self.target_step):
                target = (external_it_budget_w if external_it_budget_w is not None else
                          self.targets.power_w[time // self.targets.update_interval_seconds])
                self.interval_targets.append(target)
        self.has_external_budget |= external_it_budget_w is not None

    def intervals(self):
        for i in range(0, len(self.rows), 3):
            yield PowerInterval(self.run_id, int(self.rows[i]), int(self.rows[i + 1]), self.rows[i + 2])

    def publish(self, output_dir, episode_id, random_seed, *, it_tariff=None):
        targets = self.targets
        if self.has_external_budget:
            # A variable externally allocated IT budget has no native IT P/R
            # contract. Reuse S2 evaluation with R=0 (normalized score is N/A).
            observed_end = int(self.rows[-2]) if self.rows else 0
            targets = (TargetSchedule(self.run_id, tuple(self.interval_targets),
                                      self.target_step, 0, observed_end, 0.) if observed_end else None)
        path = Path(output_dir).resolve() / 'tracking' / (self.run_id + '.json')
        summary = dict(report_version=REPORT_VERSION, metric_version=METRIC_VERSION,
                       run_id=self.run_id, episode_id=episode_id, artifact_path=str(path),
                       power_boundary='IT', power_unit='W', time_unit='s',
                       configured_duration_seconds=self.duration,
                       observed_end_seconds=int(self.rows[-2]) if self.rows else 0,
                       source_interval_count=len(self.rows) // 3,
                       baseline_power_w=self.baseline_power_w, random_seed=int(random_seed),
                       legacy=dict(metric_version=LEGACY_VERSION,
                                   tracking_file='tracking_error_90th.csv',
                                   simulator_attribute='tracking_error_at_90',
                                   run_return='legacy_power_cost_and_tracking_cost',
                                   sa_objective='unchanged_legacy_objective'))
        if self.has_external_budget:
            summary.update(target_source='external_it_budget_with_native_fallback',
                           baseline_power_w=None)
            summary['legacy']['status'] = 'not_applicable_to_external_it_budget'
        if self.duration == 0 or self.duration % self.dt:
            # Preserve the native sample grid: never invent a final partial step
            # or present a shorter window as a complete experiment.
            summary.update(status=('unavailable_no_elapsed_time' if self.duration == 0
                                   else 'unavailable_incomplete_window'), metrics=None,
                           paper_reference=None)
        else:
            result = evaluate_tracking(self.intervals(), targets,
                                       window_start_seconds=0, window_end_seconds=self.duration)
            reference = paper_reference(result)
            result.pop('segments')
            summary.update(result, status='complete', paper_reference=reference)
        energy_cost = evaluate_it_energy_cost(
            self.intervals(), run_id=self.run_id, window_start_seconds=0,
            window_end_seconds=summary['observed_end_seconds'], tariff=it_tariff)
        energy_cost.update(episode_id=episode_id, artifact_path=str(path),
                           configured_duration_seconds=self.duration)
        if summary['observed_end_seconds'] != self.duration:
            # These values describe only the observed interval, never a guessed
            # full-window bill. The existing native sample grid is unchanged.
            energy_cost['status'] = 'incomplete_window'
            if it_tariff is not None:
                energy_cost['it_electricity_cost_estimate']['status'] = 'partial_estimate'
        payload = dict(schema_version=REPORT_VERSION, summary=summary,
                       energy_cost=energy_cost,
                       inputs=dict(target_schedule=asdict(targets) if targets else None,
                                   interval_columns=['start_seconds', 'end_seconds', 'actual_it_power_w'],
                                   power_intervals=[list(self.rows[i:i + 3])
                                                    for i in range(0, len(self.rows), 3)]))
        write_report(path, payload)
        self.energy_cost_report = energy_cost
        return summary
