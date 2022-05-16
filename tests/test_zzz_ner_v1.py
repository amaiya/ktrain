#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
import os
from unittest import TestCase, main, skip

import IPython
import numpy as np
import testenv

os.environ["DISABLE_V2_BEHAVIOR"] = "1"

import ktrain
from ktrain import text as txt


class TestNERClassification(TestCase):
    def setUp(self):
        TDATA = "resources/conll2003/train.txt"
        (trn, val, preproc) = txt.entities_from_txt(TDATA, use_char=True)
        self.trn = trn
        self.val = val
        self.preproc = preproc

    def test_ner(self):
        wv_url = (
            "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
        )
        model = txt.sequence_tagger("bilstm-crf", self.preproc, wv_path_or_url=wv_url)
        learner = ktrain.get_learner(
            model, train_data=self.trn, val_data=self.val, batch_size=128
        )
        lr = 0.01
        hist = learner.fit(lr, 1)

        # test training results
        # self.assertAlmostEqual(max(hist.history['lr']), lr)
        self.assertGreater(learner.validate(), 0.65)

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


if __name__ == "__main__":
    main()
