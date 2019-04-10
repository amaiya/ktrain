#!/usr/bin/env python3
"""
Tests of ktrain image classification flows
"""
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.insert(0,'../..')
from unittest import TestCase, main, skip
import ktrain
from ktrain import vision as vis

def classify_from_folder():
    (trn, val, preproc) = vis.images_from_folder(
                                                  datadir='image_data/image_folder',
                                                  data_aug=vis.get_data_aug(horizontal_flip=True), 
                                                  train_test_names=['train', 'valid'])
    model = vis.image_classifier('pretrained_resnet50', trn, val)
    learner = ktrain.get_learner(model=model, train_data=trn, val_data=val, batch_size=1)
    learner.freeze()
    hist = learner.autofit(1e-3, 10)
    return hist

def classify_from_csv():
    train_fpath = './image_data/train-vision.csv'
    val_fpath = './image_data/valid-vision.csv'
    trn, val, preproc = vis.images_from_csv(
                          train_fpath,
                          'filename',
                          directory='./image_data/image_folder/all',
                          val_filepath = val_fpath,
                          label_columns = ['cat', 'dog'], 
                          data_aug=vis.get_data_aug(horizontal_flip=True))
    print(vars(trn))
    model = vis.image_classifier('pretrained_resnet50', trn, val)
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=1)
    learner.freeze()
    hist = learner.autofit(1e-3, 10)
    return hist

class TestImageClassification(TestCase):
    def test_folder(self):
        hist  = classify_from_folder()
        acc = max(hist.history['acc'])
        self.assertGreater(acc, 0.8)
    def test_csv(self):
        hist  = classify_from_csv()
        acc = max(hist.history['acc'])
        self.assertGreater(acc, 0.8)

if __name__ == "__main__":
    main()

