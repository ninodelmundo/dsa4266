import math
import unittest

import numpy as np
import pandas as pd

from src.explainability.constants import HTML_FEATURE_NAMES, URL_FEATURE_NAMES
from src.explainability.engine import build_dashboard_frame, dashboard_feature_columns


class ExplainabilityUtilityTests(unittest.TestCase):
    def test_feature_names_are_stable(self):
        self.assertEqual(len(URL_FEATURE_NAMES), 9)
        self.assertIn("url_has_ip", URL_FEATURE_NAMES)
        self.assertEqual(len(HTML_FEATURE_NAMES), 8)
        self.assertIn("html_has_password", HTML_FEATURE_NAMES)

    def test_dashboard_frame_contains_interpretable_inputs(self):
        sample_df = pd.DataFrame(
            {
                "feature_index": [0],
                "split": ["test"],
                "url": ["https://login.example.com/account?verify=1"],
                "html_content": ['<form><input type="password"></form>'],
            }
        )
        features = {"labels": np.array([1])}
        import torch

        features["labels"] = torch.tensor([1])
        frame = build_dashboard_frame(
            sample_df,
            features,
            fusion_probs=np.array([0.8]),
            unimodal_probs={
                "url": np.array([0.7]),
                "text": np.array([0.6]),
                "visual": np.array([0.4]),
                "html": np.array([0.9]),
            },
        )
        self.assertEqual(frame.loc[0, "predicted_label"], 1)
        self.assertEqual(frame.loc[0, "num_modalities_above_threshold"], 3)
        self.assertGreater(frame.loc[0, "disagreement_entropy"], 0)
        self.assertIn("url_phishing_prob", dashboard_feature_columns(frame))
        self.assertNotIn("fusion_phishing_prob", dashboard_feature_columns(frame))

    def test_modality_group_shapley_identity_case(self):
        # For f(S) = sum(weights for present modalities), Shapley recovers weights.
        modalities = ["url", "text", "visual"]
        weights = {"url": 0.2, "text": 0.5, "visual": -0.1}

        def value(coalition):
            return sum(weights[name] for name in coalition)

        recovered = {}
        m = len(modalities)
        for modality in modalities:
            phi = 0.0
            others = [name for name in modalities if name != modality]
            for size in range(m):
                from itertools import combinations

                for subset in combinations(others, size):
                    subset = set(subset)
                    coef = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
                    phi += coef * (value(subset | {modality}) - value(subset))
            recovered[modality] = phi

        for modality, expected in weights.items():
            self.assertAlmostEqual(recovered[modality], expected)


if __name__ == "__main__":
    unittest.main()
