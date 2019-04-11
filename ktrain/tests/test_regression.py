#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.insert(0,'../..')
from unittest import TestCase, main, skip
import ktrain

def bostonhousing():
    from keras.datasets import boston_housing
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    from keras.models import Sequential
    from keras.layers import Dense 
    model = Sequential()
    model.add(Dense(1, input_shape=(x_train.shape[1],), activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))
    learner.lr_find()
    hist = learner.fit(0.05, 8, cycle_len=1, cycle_mult=2)
    learner.view_top_losses(n=1)
    return hist



class TestRegression(TestCase):

    def test_bostonhousing(self):
        hist  = bostonhousing()
        min_loss = min(hist.history['val_loss'])
        print(min_loss)
        self.assertLess(min_loss, 40)

if __name__ == "__main__":
    main()
