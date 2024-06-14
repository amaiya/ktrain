#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
from unittest import TestCase, main, skip

import IPython
import numpy as np
import testenv

import ktrain
from ktrain import text as txt
from ktrain.imports import ACC_NAME, VAL_ACC_NAME

TEST_DOC = "god christ jesus mother mary church sunday lord heaven amen"
EVAL_BS = 64


class TestTransformers(TestCase):
    def setUp(self):

        # Fetching data
        from sklearn.datasets import fetch_20newsgroups
        import pandas as pd
        import numpy as np

        classes = ["soc.religion.christian", "sci.space"]
        newsgroups = fetch_20newsgroups(subset="all", categories=classes)
        corpus, group_labels = (
            np.array(newsgroups.data),
            np.array(newsgroups.target_names)[newsgroups.target],
        )

        # Wrangling data into a dataframe and selecting training examples
        data = pd.DataFrame({"text": corpus, "label": group_labels})
        train_df = data.groupby("label").sample(500)
        test_df = data.drop(index=train_df.index)

        x_train = train_df["text"].values
        y_train = train_df["label"].values
        x_test = test_df["text"].values
        y_test = test_df["label"].values

        # setup
        self.trn = (x_train, y_train)
        self.val = (x_test, y_test)
        # self.classes = train_b.target_names
        self.classes = []  # discover from string labels

    # @skip("temporarily disabled")
    def test_textclassification(self):
        trn, val, preproc = txt.texts_from_array(
            x_train=self.trn[0],
            y_train=self.trn[1],
            x_test=self.val[0],
            y_test=self.val[1],
            class_names=self.classes,
            preprocess_mode="distilbert",
            maxlen=350,
        )
        model = txt.text_classifier("distilbert", train_data=trn, preproc=preproc)
        learner = ktrain.get_learner(
            model,
            train_data=trn,
            val_data=val,
            batch_size=6,
            # eval_batch_size=EVAL_BS
        )

        # test weight decay
        # NOTE due to transformers and/or AdamW bug, # val_accuracy is missing in training history if setting weight decay prior to training
        # self.assertEqual(learner.get_weight_decay(), None)
        # learner.set_weight_decay(1e-2)
        # self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # train
        lr = 5e-5
        hist = learner.fit_onecycle(lr, 1)

        # test training results
        self.assertAlmostEqual(max(hist.history["lr"]), lr)
        self.assertGreater(max(hist.history[VAL_ACC_NAME]), 0.9)

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
        learner.load_model(tmp_folder)

        # test validate
        cm = learner.validate()
        print(cm)
        for i, row in enumerate(cm):
            self.assertEqual(np.argmax(row), i)

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc, batch_size=EVAL_BS)
        self.assertEqual(p.predict([TEST_DOC])[0], "soc.religion.christian")
        tmp_folder = ktrain.imports.tempfile.mkdtemp()
        p.save(tmp_folder)
        p = ktrain.load_predictor(tmp_folder, batch_size=EVAL_BS)
        self.assertEqual(p.predict(TEST_DOC), "soc.religion.christian")
        self.assertEqual(np.argmax(p.predict_proba([TEST_DOC])[0]), 1)
        self.assertEqual(type(p.explain(TEST_DOC)), IPython.core.display.HTML)

    # @skip('temporarily disabled')
    def test_textregression(self):
        MODEL_NAME = "distilbert-base-uncased"
        x_train = self.trn[0]
        y_train = self.trn[1]
        x_test = self.val[0]
        y_test = self.val[1]
        y_train = [1.0 if y == "soc.religion.christian" else 0.0 for y in y_train]
        y_test = [1.0 if y == "soc.religion.christian" else 0.0 for y in y_test]

        preproc = txt.Transformer(MODEL_NAME, maxlen=350, classes=self.classes)
        trn = preproc.preprocess_train(x_train, y_train)
        val = preproc.preprocess_test(x_test, y_test)
        model = preproc.get_regression_model()
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
        lr = 5e-5
        hist = learner.fit_onecycle(lr, 1)

        # test training results
        self.assertAlmostEqual(max(hist.history["lr"]), lr)
        self.assertLess(min(hist.history["val_mae"]), 0.1)

        # test top losses
        obs = learner.top_losses(n=1, val_data=None)
        self.assertIn(obs[0][0], list(range(len(y_test))))
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
        self.assertGreater(p.predict([TEST_DOC])[0], 0.9)
        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        self.assertGreater(p.predict([TEST_DOC])[0], 0.9)
        self.assertIsNone(p.explain(TEST_DOC))


if __name__ == "__main__":
    main()
