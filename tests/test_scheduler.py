import numpy as np

from dspark_opd.scheduler import (
    expected_throughput,
    hardware_aware_prefix_scheduler,
    survival_probabilities,
)


def test_survival_probabilities_are_cumulative_products():
    confidence = np.array([[0.9, 0.8, 0.5]])
    out = survival_probabilities(confidence)
    assert np.allclose(out, np.array([[0.9, 0.72, 0.36]]))


def test_scheduler_keeps_high_return_prefixes():
    confidence = np.array(
        [
            [0.9, 0.8, 0.7],
            [0.3, 0.3, 0.3],
        ]
    )
    sps = {2: 1.0, 3: 0.95, 4: 0.85, 5: 0.60, 6: 0.45, 7: 0.30, 8: 0.20}
    schedule = hardware_aware_prefix_scheduler(confidence, sps)
    assert schedule.lengths[0] >= 1
    assert schedule.lengths[0] >= schedule.lengths[1]
    throughput, expected_accepts, batch_size = expected_throughput(schedule.lengths, survival_probabilities(confidence), sps)
    assert np.isclose(schedule.best_throughput, throughput)
    assert np.isclose(schedule.expected_accepts, expected_accepts)
    assert schedule.batch_size == batch_size


def test_scheduler_can_run_without_early_stop_for_async_capacity_mode():
    confidence = np.array([[0.8, 0.8], [0.7, 0.7]])
    sps = {2: 1.0, 3: 0.8, 4: 0.9, 5: 0.4, 6: 0.3}
    early = hardware_aware_prefix_scheduler(confidence, sps, early_stop=True)
    globalish = hardware_aware_prefix_scheduler(confidence, sps, early_stop=False)
    assert globalish.best_throughput >= early.best_throughput
