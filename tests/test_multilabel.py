#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
from unittest import TestCase, main, skip

import numpy as np
import testenv

import ktrain
from ktrain import utils as U
from ktrain.imports import ACC_NAME, VAL_ACC_NAME

Sequential = ktrain.imports.keras.models.Sequential
Dense = ktrain.imports.keras.layers.Dense
Embedding = ktrain.imports.keras.layers.Embedding
GlobalAveragePooling1D = ktrain.imports.keras.layers.GlobalAveragePooling1D


def synthetic_multilabel():
    # data
    X = [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0],
        [3, 4, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0, 0],
        [1, 2, 3, 0, 0, 0, 0],
        [1, 2, 3, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 2, 3, 0, 0, 0],
        [2, 3, 3, 4, 0, 0, 0],
        [4, 4, 1, 1, 2, 0, 0],
        [1, 2, 3, 3, 3, 3, 3],
        [2, 4, 2, 4, 2, 0, 0],
        [1, 3, 3, 3, 0, 0, 0],
        [4, 4, 0, 0, 0, 0, 0],
        [3, 3, 0, 0, 0, 0, 0],
        [1, 1, 4, 0, 0, 0, 0],
    ]

    Y = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
    ]

    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)


class TestMultilabel(TestCase):
    def test_multilabel(self):
        X, Y = synthetic_multilabel()
        self.assertTrue(U.is_multilabel((X, Y)))
        MAXLEN = 7
        MAXFEATURES = 4
        NUM_CLASSES = 4
        model = Sequential()
        model.add(Embedding(MAXFEATURES + 1, 50, input_length=MAXLEN))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(NUM_CLASSES, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        learner = ktrain.get_learner(
            model, train_data=(X, Y), val_data=(X, Y), batch_size=1
        )
        learner.lr_find(max_epochs=5)  # use max_epochs until TF 2.4

        # use loss instead of accuracy due to: https://github.com/tensorflow/tensorflow/issues/41114
        hist = learner.fit(0.001, 200)
        learner.view_top_losses(n=5)
        learner.validate()
        # final_acc = hist.history[VAL_ACC_NAME][-1]
        # print('final_accuracy:%s' % (final_acc))
        # self.assertGreater(final_acc, 0.97)

        final_loss = hist.history["val_loss"][-1]
        print("final_loss:%s" % (final_loss))
        self.assertLess(final_loss, 0.05)


if __name__ == "__main__":
    main()
