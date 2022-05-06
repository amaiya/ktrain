#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
from unittest import TestCase, main, skip

import testenv

import ktrain

Sequential = ktrain.imports.keras.models.Sequential
Dense = ktrain.imports.keras.layers.Dense


def bostonhousing():
    from tensorflow.keras.datasets import boston_housing

    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    model = Sequential()
    model.add(Dense(1, input_shape=(x_train.shape[1],), activation="linear"))
    model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
    learner = ktrain.get_learner(
        model, train_data=(x_train, y_train), val_data=(x_test, y_test)
    )
    learner.lr_find(max_epochs=5)  # use max_epochs until TF 2.4
    hist = learner.fit(0.05, 8, cycle_len=1, cycle_mult=2)
    learner.view_top_losses(n=5)
    learner.validate()
    return hist


class TestRegression(TestCase):
    def test_bostonhousing(self):
        hist = bostonhousing()
        min_loss = min(hist.history["val_loss"])
        print(min_loss)
        self.assertLess(min_loss, 55)


if __name__ == "__main__":
    main()
