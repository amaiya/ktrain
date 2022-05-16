#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
from unittest import TestCase, main, skip

import numpy as np
import testenv

from ktrain import tabular


def adult_census():
    import pandas as pd

    df = pd.read_csv("resources/tabular_data/adults.csv")
    df = df.rename(columns=lambda x: x.strip())
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    filter_set = "Doctorate"
    df["treatment"] = df["education"].apply(lambda x: 1 if x in filter_set else 0)
    return df


class TestCausalInference(TestCase):
    def test_ci(self):
        df = adult_census()
        cm = tabular.causalinference.causal_inference_model(
            df,
            metalearner_type="t-learner",
            treatment_col="treatment",
            outcome_col="class",
            ignore_cols=["fnlwgt", "education", "education-num"],
        ).fit()
        ate = cm.estimate_ate()
        self.assertGreater(ate["ate"], 0.20)
        self.assertLess(ate["ate"], 0.21)


if __name__ == "__main__":
    main()
