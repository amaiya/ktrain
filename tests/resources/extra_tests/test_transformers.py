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
        # fetch the dataset using scikit-learn
        categories = [
            "alt.atheism",
            "soc.religion.christian",
            "comp.graphics",
            "sci.med",
        ]
        from sklearn.datasets import fetch_20newsgroups

        train_b = fetch_20newsgroups(
            subset="train", categories=categories, shuffle=True, random_state=42
        )
        test_b = fetch_20newsgroups(
            subset="test", categories=categories, shuffle=True, random_state=42
        )
        print("size of training set: %s" % (len(train_b["data"])))
        print("size of validation set: %s" % (len(test_b["data"])))
        print("classes: %s" % (train_b.target_names))
        x_train = train_b.data
        y_train = train_b.target
        x_test = test_b.data
        y_test = test_b.target

        # convert to string labels
        y_train = [train_b.target_names[y] for y in y_train]
        y_test = [train_b.target_names[y] for y in y_test]

        # setup
        self.trn = (x_train, y_train)
        self.val = (x_test, y_test)
        # self.classes = train_b.target_names
        self.classes = []  # discover from string labels

    # @skip("temporarily disabled")
    def test_transformers_api_1(self):
        trn, val, preproc = txt.texts_from_array(
            x_train=self.trn[0],
            y_train=self.trn[1],
            x_test=self.val[0],
            y_test=self.val[1],
            class_names=self.classes,
            preprocess_mode="distilbert",
            maxlen=500,
            max_features=35000,
        )
        model = txt.text_classifier("distilbert", train_data=trn, preproc=preproc)
        learner = ktrain.get_learner(
            model, train_data=trn, val_data=val, batch_size=6, eval_batch_size=EVAL_BS
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
        self.assertEqual(np.argmax(p.predict_proba([TEST_DOC])[0]), 3)
        self.assertEqual(type(p.explain(TEST_DOC)), IPython.core.display.HTML)

    # @skip('temporarily disabled')
    def test_transformers_api_2(self):
        MODEL_NAME = "distilbert-base-uncased"
        preproc = txt.Transformer(MODEL_NAME, maxlen=500, classes=self.classes)
        trn = preproc.preprocess_train(self.trn[0], self.trn[1])
        val = preproc.preprocess_test(self.val[0], self.val[1])
        model = preproc.get_classifier()
        learner = ktrain.get_learner(
            model, train_data=trn, val_data=val, batch_size=6, eval_batch_size=EVAL_BS
        )
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
        self.assertEqual(np.argmax(p.predict_proba([TEST_DOC])[0]), 3)
        self.assertEqual(type(p.explain(TEST_DOC)), IPython.core.display.HTML)


if __name__ == "__main__":
    main()
