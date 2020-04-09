#!/usr/bin/env python3
"""
Tests of ktrain image classification flows
"""
import testenv
from unittest import TestCase, main, skip
import numpy as np



import ktrain
from ktrain import vision as vis
import ktrain.utils as U
from ktrain.imports import ACC_NAME, VAL_ACC_NAME


#def classify_from_csv():
    #train_fpath = './image_data/train-vision.csv'
    #val_fpath = './image_data/valid-vision.csv'
    #trn, val, preproc = vis.images_from_csv(
                          #train_fpath,
                          #'filename',
                          #directory='./image_data/image_folder/all',
                          #val_filepath = val_fpath,
                          #label_columns = ['cat', 'dog'], 
                          #data_aug=vis.get_data_aug(horizontal_flip=True))
    #print(vars(trn))
    #model = vis.image_classifier('pretrained_resnet50', trn, val)
    #learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=1)
    #learner.freeze()
    #hist = learner.autofit(1e-3, 10)
    #return hist


class TestImageClassification(TestCase):
    #@skip('temporarily disabled')
    def test_folder(self):
        (trn, val, preproc) = vis.images_from_folder(
                                                      datadir='image_data/image_folder',
                                                      data_aug=vis.get_data_aug(horizontal_flip=True), 
                                                      classes=['cat', 'dog'],
                                                      train_test_names=['train', 'valid'])
        model = vis.image_classifier('pretrained_resnet50', trn, val)
        learner = ktrain.get_learner(model=model, train_data=trn, val_data=val, batch_size=1)
        learner.freeze()
        hist = learner.autofit(1e-3, monitor=VAL_ACC_NAME)

        # test train
        self.assertAlmostEqual(max(hist.history['lr']), 1e-3)
        if max(hist.history[ACC_NAME]) == 0.5:
            raise Exception('unlucky initialization: please run test again')
        self.assertGreater(max(hist.history[ACC_NAME]), 0.8)

        # test top_losses
        obs = learner.top_losses(n=1, val_data=val)
        print(obs)
        if obs:
            self.assertIn(obs[0][0], list(range(U.nsamples_from_data(val))))
        else:
            self.assertEqual(max(hist.history[VAL_ACC_NAME]), 1)

        # test weight decay
        self.assertEqual(learner.get_weight_decay(), None)
        learner.set_weight_decay(1e-2)
        self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # test load and save model
        learner.save_model('/tmp/test_model')
        learner.load_model('/tmp/test_model')

        # test validate
        cm = learner.validate(val_data=val)
        print(cm)
        for i, row in enumerate(cm):
            self.assertEqual(np.argmax(row), i)

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc)
        r = p.predict_folder('image_data/image_folder/train/')
        print(r)
        self.assertEqual(r[0][1], 'cat')
        r = p.predict_proba_folder('image_data/image_folder/train/')
        self.assertEqual(np.argmax(r[0][1]), 0)
        r = p.predict_filename('image_data/image_folder/train/cat/cat.11737.jpg')
        self.assertEqual(r, ['cat'])
        r = p.predict_proba_filename('image_data/image_folder/train/cat/cat.11737.jpg')
        self.assertEqual(np.argmax(r), 0)

        p.save('/tmp/test_predictor')
        p = ktrain.load_predictor('/tmp/test_predictor')
        r = p.predict_filename('image_data/image_folder/train/cat/cat.11737.jpg')
        self.assertEqual(r, ['cat'])


    @skip('temporarily disabled')
    def test_csv(self):
        train_fpath = './image_data/train-vision.csv'
        val_fpath = './image_data/valid-vision.csv'
        trn, val, preproc = vis.images_from_csv(
                              train_fpath,
                              'filename',
                              directory='./image_data/image_folder/all',
                              val_filepath = val_fpath,
                              label_columns = ['cat', 'dog'], 
                              data_aug=vis.get_data_aug(horizontal_flip=True))

        lr = 1e-4
        model = vis.image_classifier('pretrained_resnet50', trn, val)
        learner = ktrain.get_learner(model=model, train_data=trn, val_data=val, batch_size=4)
        learner.freeze()
        hist = learner.fit_onecycle(lr, 3)

        # test train
        self.assertAlmostEqual(max(hist.history['lr']), lr)
        if max(hist.history[ACC_NAME]) == 0.5:
            raise Exception('unlucky initialization: please run test again')
        self.assertGreater(max(hist.history[ACC_NAME]), 0.8)

        # test top_losses
        obs = learner.top_losses(n=1, val_data=val)
        print(obs)
        if obs:
            self.assertIn(obs[0][0], list(range(U.nsamples_from_data(val))))
        else:
            self.assertEqual(max(hist.history[VAL_ACC_NAME]), 1)

        # test weight decay
        self.assertEqual(learner.get_weight_decay(), None)
        learner.set_weight_decay(1e-2)
        self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # test load and save model
        learner.save_model('/tmp/test_model')
        learner.load_model('/tmp/test_model')

        # test validate
        cm = learner.validate(val_data=val)
        print(cm)
        for i, row in enumerate(cm):
            self.assertEqual(np.argmax(row), i)

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc)
        r = p.predict_folder('image_data/image_folder/train/')
        print(r)
        self.assertEqual(r[0][1], 'cat')
        r = p.predict_proba_folder('image_data/image_folder/train/')
        self.assertEqual(np.argmax(r[0][1]), 0)
        r = p.predict_filename('image_data/image_folder/train/cat/cat.11737.jpg')
        self.assertEqual(r, ['cat'])
        r = p.predict_proba_filename('image_data/image_folder/train/cat/cat.11737.jpg')
        self.assertEqual(np.argmax(r), 0)

        p.save('/tmp/test_predictor')
        p = ktrain.load_predictor('/tmp/test_predictor')
        r = p.predict_filename('image_data/image_folder/train/cat/cat.11737.jpg')
        self.assertEqual(r, ['cat'])



if __name__ == "__main__":
    main()

