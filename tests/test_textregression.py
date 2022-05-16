#!/usr/bin/env python3
"""
Tests of ktrain text regression
"""
from unittest import TestCase, main, skip

import IPython
import numpy as np
import testenv

import ktrain
from ktrain import text as txt
from ktrain.imports import ACC_NAME, VAL_ACC_NAME

TEST_DOC = """A wine that has created its own universe. It has a unique, special softness
              that allies with the total purity that comes from a small, enclosed single vineyard.
              The fruit is almost irrelevant here, because it comes as part of a much deeper complexity.
              This is a great wine, at the summit of Champagne, a sublime, unforgettable experience.
              """


class TestTextRegression(TestCase):
    def setUp(self):
        import pandas as pd

        # wine price dataset should be downloaded
        # from: https://github.com/floydhub/regression-template
        # and prepared as described in the wide-deep.ipynb notebook
        path = "./resources/text_data/wines.csv"
        data = pd.read_csv(path)
        data = data.sample(frac=1.0, random_state=42)

        # Split data into train and test
        train_size = int(len(data) * 0.8)
        print("Train size: %d" % train_size)
        print("Test size: %d" % (len(data) - train_size))

        # Train features
        description_train = data["description"][:train_size]

        # Train labels
        labels_train = data["price"][:train_size]

        # Test features
        description_test = data["description"][train_size:]

        # Test labels
        labels_test = data["price"][train_size:]

        # dataset
        x_train = description_train.values
        y_train = labels_train.values
        x_test = description_test.values
        y_test = labels_test.values
        self.trn = (x_train, y_train)
        self.val = (x_test, y_test)

    # @skip('temporarily disabled')
    def test_linreg(self):
        trn, val, preproc = txt.texts_from_array(
            x_train=self.trn[0],
            y_train=self.trn[1],
            x_test=self.val[0],
            y_test=self.val[1],
            preprocess_mode="standard",
            ngram_range=3,
            maxlen=200,
            max_features=35000,
        )
        model = txt.text_regression_model("linreg", train_data=trn, preproc=preproc)
        learner = ktrain.get_learner(
            model, train_data=trn, val_data=val, batch_size=256
        )
        lr = 0.01
        hist = learner.fit_onecycle(lr, 10)

        # test training results
        self.assertAlmostEqual(max(hist.history["lr"]), lr)
        self.assertLess(min(hist.history["val_mae"]), 12)

        # test top losses
        obs = learner.top_losses(n=1, val_data=None)
        self.assertIn(obs[0][0], list(range(len(val[0]))))
        learner.view_top_losses(preproc=preproc, n=1, val_data=None)

        # test weight decay
        self.assertEqual(learner.get_weight_decay(), None)
        learner.set_weight_decay(1e-2)
        self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # test load and save model
        learner.save_model("/tmp/test_model")
        learner.load_model("/tmp/test_model")

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc)
        self.assertGreater(p.predict([TEST_DOC])[0], 100)
        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        self.assertGreater(p.predict([TEST_DOC])[0], 100)
        self.assertIsNone(p.explain(TEST_DOC))

    # @skip('temporarily disabled')
    def test_distilbert(self):
        trn, val, preproc = txt.texts_from_array(
            x_train=self.trn[0],
            y_train=self.trn[1],
            x_test=self.val[0],
            y_test=self.val[1],
            preprocess_mode="distilbert",
            maxlen=75,
        )
        model = txt.text_regression_model("distilbert", train_data=trn, preproc=preproc)
        learner = ktrain.get_learner(
            model, train_data=trn, val_data=val, batch_size=100
        )
        lr = 5e-5
        hist = learner.fit_onecycle(lr, 1)

        # test training results
        self.assertAlmostEqual(max(hist.history["lr"]), lr)
        self.assertLess(min(hist.history["val_mae"]), 16)

        # test top losses
        obs = learner.top_losses(n=1, val_data=None)
        self.assertIn(obs[0][0], list(range(len(val.x))))
        learner.view_top_losses(preproc=preproc, n=1, val_data=None)

        # test weight decay
        self.assertEqual(learner.get_weight_decay(), None)
        learner.set_weight_decay(1e-2)
        self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # test load and save model
        tmp_folder = ktrain.imports.tempfile.mkdtemp()
        learner.save_model(tmp_folder)
        learner.load_model(tmp_folder, preproc=preproc)

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc, batch_size=64)
        self.assertGreater(p.predict([TEST_DOC])[0], 1)
        tmp_folder = ktrain.imports.tempfile.mkdtemp()
        p.save(tmp_folder)
        p = ktrain.load_predictor(tmp_folder, batch_size=64)
        self.assertGreater(p.predict([TEST_DOC])[0], 1)
        self.assertIsNone(p.explain(TEST_DOC))


if __name__ == "__main__":
    main()
