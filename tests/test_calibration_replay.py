import numpy as np

from dspark_opd.calibration import expected_calibration_error, sequential_temperature_scaling
from dspark_opd.replay import position_decay_weights, split_acceptance_masks


def test_position_decay_weights():
    weights = position_decay_weights(4, gamma=0.5)
    assert np.allclose(weights, np.array([1.0, 0.5, 0.25, 0.125]))


def test_split_acceptance_masks():
    accepted, rejected = split_acceptance_masks(np.array([0, 2]), block_size=3)
    assert np.allclose(accepted[0], np.array([0, 0, 0]))
    assert np.allclose(rejected[0], np.array([1, 1, 1]))
    assert np.allclose(accepted[1], np.array([1, 1, 0]))
    assert np.allclose(rejected[1], np.array([0, 0, 1]))


def test_sequential_temperature_scaling_reduces_or_matches_ece():
    confidence = np.array(
        [
            [0.95, 0.90],
            [0.90, 0.80],
            [0.75, 0.60],
            [0.60, 0.40],
            [0.40, 0.20],
            [0.20, 0.10],
        ]
    )
    labels = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    result = sequential_temperature_scaling(confidence, labels, grid=np.linspace(0.5, 2.5, 11), n_bins=3)
    assert result.temperatures.shape == (2,)
    assert np.all(result.after_ece <= result.before_ece + 1e-12)
    assert expected_calibration_error(confidence[:, 0], labels[:, 0], n_bins=3) >= result.after_ece[0]
