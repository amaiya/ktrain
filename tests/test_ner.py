#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
import os
from unittest import TestCase, main, skip

import IPython
import numpy as np
import testenv

os.environ["DISABLE_V2_BEHAVIOR"] = "0"

import ktrain
from ktrain import text as txt


class TestNERClassification(TestCase):
    def setUp(self):
        TDATA = "resources/conll2003/train.txt"
        (trn, val, preproc) = txt.entities_from_txt(TDATA)
        self.trn = trn
        self.val = val
        self.preproc = preproc

    def test_ner(self):
        model = txt.sequence_tagger(
            "bilstm-transformer", self.preproc, transformer_model="roberta-base"
        )
        learner = ktrain.get_learner(
            model, train_data=self.trn, val_data=self.val, batch_size=128
        )
        lr = 0.01
        hist = learner.fit(lr, 1)

        # test training results
        # self.assertAlmostEqual(max(hist.history['lr']), lr)
        self.assertGreater(learner.validate(), 0.79)

        # test top losses
        obs = learner.top_losses(n=1)
        self.assertIn(obs[0][0], list(range(len(self.val.x))))
        learner.view_top_losses(n=1)

        # test weight decay
        self.assertEqual(learner.get_weight_decay(), None)
        learner.set_weight_decay(1e-2)
        self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # test load and save model
        learner.save_model("/tmp/test_model")
        learner.load_model("/tmp/test_model")

        # test predictor
        SENT = "There is a man named John Smith."
        p = ktrain.get_predictor(learner.model, self.preproc)
        self.assertEqual(p.predict(SENT)[-2][1], "I-PER")
        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        self.assertEqual(p.predict(SENT)[-2][1], "I-PER")
        merged_prediction = p.predict(SENT, merge_tokens=True, return_offsets=True)
        self.assertEqual(merged_prediction[0][0], "John Smith")
        self.assertEqual(merged_prediction[0][1], "PER")
        self.assertEqual(merged_prediction[0][2], (21, 31))


if __name__ == "__main__":
    main()
