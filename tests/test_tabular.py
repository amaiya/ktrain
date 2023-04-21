#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
from unittest import TestCase, main, skip

import IPython
import numpy as np
import pandas as pd
import testenv

import ktrain
from ktrain import tabular
from ktrain.imports import ACC_NAME, VAL_ACC_NAME


class TestTabular(TestCase):
    def test_classification(self):
        train_df = pd.read_csv("resources/tabular_data/train.csv", index_col=0)
        train_df = train_df.drop("Name", axis=1)
        train_df = train_df.drop("Ticket", axis=1)
        trn, val, preproc = tabular.tabular_from_df(
            train_df, label_columns="Survived", random_state=42
        )
        model = tabular.tabular_classifier("mlp", trn)
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=32)

        lr = 0.001
        hist = learner.fit_onecycle(lr, 30)

        # test training results
        self.assertAlmostEqual(max(hist.history["lr"]), lr)
        self.assertGreater(max(hist.history[VAL_ACC_NAME]), 0.8)

        # test top losses
        obs = learner.top_losses(n=1, val_data=val)
        self.assertIn(obs[0][0], list(range(val.df.shape[0])))
        learner.view_top_losses(preproc=preproc, n=1, val_data=val)

        # test weight decay
        self.assertEqual(learner.get_weight_decay(), None)
        learner.set_weight_decay(1e-2)
        self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # test load and save model
        learner.save_model("/tmp/test_model")
        learner.load_model("/tmp/test_model")

        # test validate
        cm = learner.evaluate(val)
        print(cm)
        for i, row in enumerate(cm):
            self.assertEqual(np.argmax(row), i)

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc)

        predicted_label = p.predict(train_df)[0]
        self.assertIn(predicted_label, preproc.get_classes())
        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        self.assertEqual(p.predict(train_df)[0], predicted_label)

    def test_regression(self):
        trn, val, preproc = tabular.tabular_from_csv(
            "resources/tabular_data/adults.csv",
            label_columns=["age"],
            is_regression=True,
            random_state=42,
        )
        model = tabular.tabular_regression_model("mlp", trn)
        learner = ktrain.get_learner(
            model, train_data=trn, val_data=val, batch_size=128
        )

        lr = 0.001
        hist = learner.autofit(lr, 5)

        # test training results
        self.assertAlmostEqual(max(hist.history["lr"]), lr)
        self.assertLess(min(hist.history["val_mae"]), 8.0)

        # test top losses
        obs = learner.top_losses(n=1, val_data=val)
        self.assertIn(obs[0][0], list(range(val.df.shape[0])))
        learner.view_top_losses(preproc=preproc, n=1, val_data=val)

        # test weight decay
        self.assertEqual(learner.get_weight_decay(), None)
        learner.set_weight_decay(1e-2)
        self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # test load and save model
        learner.save_model("/tmp/test_model")
        learner.load_model("/tmp/test_model")

        # test validate
        cm = learner.evaluate(val)

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc)

        train_df = pd.read_csv("resources/tabular_data/adults.csv")
        age = p.predict(train_df)[0][0]
        self.assertLess(age, 100)
        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        self.assertAlmostEqual(p.predict(train_df)[0][0], age)


if __name__ == "__main__":
    main()
