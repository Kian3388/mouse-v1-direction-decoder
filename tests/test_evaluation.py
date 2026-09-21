import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mouse_v1_decoder import direction_to_orientation, evaluate_decoder, population_size_curve


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        labels = np.repeat([0, 45, 90, 135], 30)
        signal = np.eye(4)[np.repeat(np.arange(4), 30)] * 4
        noise = rng.normal(0, 0.5, size=(labels.size, 8))
        self.features = np.column_stack([signal, noise])
        self.labels = labels

    def test_orientation_mapping(self) -> None:
        directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        expected = np.array([0, 45, 90, 135, 0, 45, 90, 135])
        np.testing.assert_array_equal(direction_to_orientation(directions), expected)

    def test_cross_validated_decoder_finds_signal(self) -> None:
        metrics, matrix, predictions = evaluate_decoder(self.features, self.labels)
        self.assertGreater(metrics.accuracy, 0.90)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(predictions.shape, self.labels.shape)

    def test_population_curve_returns_each_repeat(self) -> None:
        rows = population_size_curve(
            self.features,
            self.labels,
            [2, 4],
            repeats=3,
        )
        self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
