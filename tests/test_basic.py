#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
import os.path
from unittest import TestCase, main, skip

import IPython
import numpy as np
import testenv

import ktrain
from ktrain import text as txt
from ktrain.imports import ACC_NAME, VAL_ACC_NAME

TEST_DOC = "还好，床很大而且很干净，前台很友好，很满意，下次还来。"

CURRDIR = os.path.dirname(__file__)


class TestTextClassification(TestCase):
    def test_fasttext_chinese(self):
        trn, val, preproc = txt.texts_from_csv(
            os.path.join(CURRDIR, "resources/text_data/chinese_hotel_reviews.csv"),
            "content",
            label_columns=["pos", "neg"],
            max_features=30000,
            maxlen=75,
            preprocess_mode="standard",
            sep="|",
        )
        model = txt.text_classifier("fasttext", train_data=trn, preproc=preproc)
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=32)
        lr = 5e-3
        hist = learner.autofit(lr, 10)

        # test training results
        self.assertAlmostEqual(max(hist.history["lr"]), lr)
        self.assertGreater(max(hist.history[VAL_ACC_NAME]), 0.85)

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

        # test validate
        cm = learner.validate(class_names=preproc.get_classes())
        print(cm)
        for i, row in enumerate(cm):
            self.assertEqual(np.argmax(row), i)

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc)
        self.assertEqual(p.predict([TEST_DOC])[0], "pos")
        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        self.assertEqual(p.predict(TEST_DOC), "pos")
        self.assertEqual(np.argmax(p.predict_proba([TEST_DOC])[0]), 0)
        self.assertEqual(type(p.explain(TEST_DOC)), IPython.core.display.HTML)
        # self.assertEqual(type(p.explain(TEST_DOC)), type(None))


if __name__ == "__main__":
    main()
