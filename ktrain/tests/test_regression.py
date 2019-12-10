#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
import testenv
from unittest import TestCase, main, skip
import ktrain
Sequential = ktrain.imports.Sequential
Dense = ktrain.imports.Dense

def bostonhousing():
    from keras.datasets import boston_housing
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    model = Sequential()
    model.add(Dense(1, input_shape=(x_train.shape[1],), activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))
    learner.lr_find()
    hist = learner.fit(0.05, 8, cycle_len=1, cycle_mult=2)
    learner.view_top_losses(n=5)
    learner.validate()
    return hist



class TestRegression(TestCase):

    def test_bostonhousing(self):
        hist  = bostonhousing()
        min_loss = min(hist.history['val_loss'])
        print(min_loss)
        self.assertLess(min_loss, 55)

if __name__ == "__main__":
    main()
