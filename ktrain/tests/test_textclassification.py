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
from ktrain import text as txt

def classify_from_folder():
    DATADIR = './text_data/text_folder'
    (x_train, y_train), (x_test, y_test), preproc = txt.texts_from_folder(DATADIR, 
                                                    max_features=100, maxlen=10, 
                                                    ngram_range=3, 
                                                    classes=['pos', 'neg'])
    model = txt.text_classifier('nbsvm', (x_train, y_train))
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=1)
    hist = learner.autofit(0.001, 250)
    return hist


def classify_from_csv():
    DATA_PATH = './text_data/texts.csv'
    (x_train, y_train), (x_test, y_test), preproc = txt.texts_from_csv(DATA_PATH,
                          'text',
                          val_filepath = DATA_PATH,
                          label_columns = ["pos", "neg"],
                          max_features=100, maxlen=10,
                          ngram_range=3)
    model = txt.text_classifier('nbsvm', (x_train, y_train))
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=1)
    hist = learner.autofit(0.001, 250)
    return hist


class TestTextClassification(TestCase):

    def test_folder(self):
        hist  = classify_from_folder()
        self.assertEqual(hist.history['val_acc'][-1], 1.0)
    def test_csv(self):
        hist  = classify_from_csv()
        self.assertEqual(hist.history['val_acc'][-1], 1.0)

if __name__ == "__main__":
    main()
