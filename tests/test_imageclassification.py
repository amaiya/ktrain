#!/usr/bin/env python3
"""
Tests of ktrain image classification flows
"""
from unittest import TestCase, main, skip

import numpy as np
import testenv

import ktrain
import ktrain.utils as U
from ktrain import vision as vis
from ktrain.imports import ACC_NAME, VAL_ACC_NAME

# def classify_from_csv():
# train_fpath = './resources/image_data/train-vision.csv'
# val_fpath = './resources/image_data/valid-vision.csv'
# trn, val, preproc = vis.images_from_csv(
# train_fpath,
#'filename',
# directory='./resources/image_data/image_folder/all',
# val_filepath = val_fpath,
# label_columns = ['cat', 'dog'],
# data_aug=vis.get_data_aug(horizontal_flip=True))
# print(vars(trn))
# model = vis.image_classifier('pretrained_resnet50', trn, val)
# learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=1)
# learner.freeze()
# hist = learner.autofit(1e-3, 10)
# return hist


class TestImageClassification(TestCase):
    # @skip('temporarily disabled')
    def test_folder(self):
        (trn, val, preproc) = vis.images_from_folder(
            datadir="resources/image_data/image_folder",
            data_aug=vis.get_data_aug(horizontal_flip=True),
            classes=["cat", "dog"],
            train_test_names=["train", "valid"],
        )
        model = vis.image_classifier("pretrained_resnet50", trn, val)
        learner = ktrain.get_learner(
            model=model, train_data=trn, val_data=val, batch_size=1
        )
        learner.freeze()

        # test weight decay
        self.assertEqual(learner.get_weight_decay(), None)
        learner.set_weight_decay(1e-2)
        self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # train
        hist = learner.autofit(1e-3, monitor=VAL_ACC_NAME)

        # test train
        self.assertAlmostEqual(max(hist.history["lr"]), 1e-3)
        if max(hist.history[ACC_NAME]) == 0.5:
            raise Exception("unlucky initialization: please run test again")
        self.assertGreater(max(hist.history[ACC_NAME]), 0.8)

        # test top_losses
        obs = learner.top_losses(n=1, val_data=val)
        print(obs)
        if obs:
            self.assertIn(obs[0][0], list(range(U.nsamples_from_data(val))))
        else:
            self.assertEqual(max(hist.history[VAL_ACC_NAME]), 1)

        # test load and save model
        learner.save_model("/tmp/test_model")
        learner.load_model("/tmp/test_model")

        # test validate
        cm = learner.validate(val_data=val)
        print(cm)
        for i, row in enumerate(cm):
            self.assertEqual(np.argmax(row), i)

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc)
        r = p.predict_folder("resources/image_data/image_folder/train/")
        print(r)
        self.assertEqual(r[0][1], "cat")
        r = p.predict_proba_folder("resources/image_data/image_folder/train/")
        self.assertEqual(np.argmax(r[0][1]), 0)
        r = p.predict_filename(
            "resources/image_data/image_folder/train/cat/cat.11737.jpg"
        )
        self.assertEqual(r, ["cat"])
        r = p.predict_proba_filename(
            "resources/image_data/image_folder/train/cat/cat.11737.jpg"
        )
        self.assertEqual(np.argmax(r), 0)

        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        r = p.predict_filename(
            "resources/image_data/image_folder/train/cat/cat.11737.jpg"
        )
        self.assertEqual(r, ["cat"])

    @skip("temporarily disabled")
    def test_csv(self):
        train_fpath = "./resources/image_data/train-vision.csv"
        val_fpath = "./resources/image_data/valid-vision.csv"
        trn, val, preproc = vis.images_from_csv(
            train_fpath,
            "filename",
            directory="./resources/image_data/image_folder/all",
            val_filepath=val_fpath,
            label_columns=["cat", "dog"],
            data_aug=vis.get_data_aug(horizontal_flip=True),
        )

        lr = 1e-4
        model = vis.image_classifier("pretrained_resnet50", trn, val)
        learner = ktrain.get_learner(
            model=model, train_data=trn, val_data=val, batch_size=4
        )
        learner.freeze()

        # test weight decay
        self.assertEqual(learner.get_weight_decay(), None)
        learner.set_weight_decay(1e-2)
        self.assertAlmostEqual(learner.get_weight_decay(), 1e-2)

        # train
        hist = learner.fit_onecycle(lr, 3)

        # test train
        self.assertAlmostEqual(max(hist.history["lr"]), lr)
        if max(hist.history[ACC_NAME]) == 0.5:
            raise Exception("unlucky initialization: please run test again")
        self.assertGreater(max(hist.history[ACC_NAME]), 0.8)

        # test top_losses
        obs = learner.top_losses(n=1, val_data=val)
        print(obs)
        if obs:
            self.assertIn(obs[0][0], list(range(U.nsamples_from_data(val))))
        else:
            self.assertEqual(max(hist.history[VAL_ACC_NAME]), 1)

        # test load and save model
        learner.save_model("/tmp/test_model")
        learner.load_model("/tmp/test_model")

        # test validate
        cm = learner.validate(val_data=val)
        print(cm)
        for i, row in enumerate(cm):
            self.assertEqual(np.argmax(row), i)

        # test predictor
        p = ktrain.get_predictor(learner.model, preproc)
        r = p.predict_folder("resources/image_data/image_folder/train/")
        print(r)
        self.assertEqual(r[0][1], "cat")
        r = p.predict_proba_folder("resources/image_data/image_folder/train/")
        self.assertEqual(np.argmax(r[0][1]), 0)
        r = p.predict_filename(
            "resources/image_data/image_folder/train/cat/cat.11737.jpg"
        )
        self.assertEqual(r, ["cat"])
        r = p.predict_proba_filename(
            "resources/image_data/image_folder/train/cat/cat.11737.jpg"
        )
        self.assertEqual(np.argmax(r), 0)

        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        r = p.predict_filename(
            "resources/image_data/image_folder/train/cat/cat.11737.jpg"
        )
        self.assertEqual(r, ["cat"])

    # @skip('temporarily disabled')
    def test_array(self):

        import numpy as np
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train /= 255
        x_test /= 255
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        classes = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        data_aug = vis.get_data_aug(
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            featurewise_center=False,
            featurewise_std_normalization=False,
        )

        (trn, val, preproc) = vis.images_from_array(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            data_aug=data_aug,
            class_names=classes,
        )

        model = vis.image_classifier("default_cnn", trn, val)
        learner = ktrain.get_learner(
            model, train_data=trn, val_data=val, batch_size=128
        )
        hist = learner.fit_onecycle(1e-3, 1)

        # test train
        self.assertAlmostEqual(max(hist.history["lr"]), 1e-3)
        self.assertGreater(max(hist.history[VAL_ACC_NAME]), 0.97)

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
        learner.save_model("/tmp/test_model")
        learner.load_model("/tmp/test_model")

        # test validate
        cm = learner.validate(val_data=val)
        print(cm)
        for i, row in enumerate(cm):
            self.assertEqual(np.argmax(row), i)

        p = ktrain.get_predictor(learner.model, preproc)
        r = p.predict(x_test[0:1])
        print(r)
        self.assertEqual(r[0], "seven")
        r = p.predict(x_test[0:1], return_proba=True)
        self.assertEqual(np.argmax(r[0]), 7)

        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        r = p.predict(x_test[0:1])
        self.assertEqual(r[0], "seven")

    # @skip('temporarily disabled')
    def test_array_regression(self):

        import numpy as np
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train /= 255
        x_test /= 255
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)

        classes = None
        data_aug = vis.get_data_aug(
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            featurewise_center=False,
            featurewise_std_normalization=False,
        )

        (trn, val, preproc) = vis.images_from_array(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            data_aug=data_aug,
            is_regression=True,
            class_names=classes,
        )

        model = vis.image_regression_model("default_cnn", trn, val)
        learner = ktrain.get_learner(
            model, train_data=trn, val_data=val, batch_size=128
        )
        hist = learner.fit_onecycle(1e-3, 1)

        # test train
        self.assertAlmostEqual(max(hist.history["lr"]), 1e-3)
        self.assertLess(max(hist.history["val_mae"]), 1)

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
        learner.save_model("/tmp/test_model")
        learner.load_model("/tmp/test_model")

        # test validate
        cm = learner.validate(val_data=val)
        print(cm)

        p = ktrain.get_predictor(learner.model, preproc)
        r = p.predict(x_test[0:1])
        print(r)
        self.assertIn(round(r[0]), [6, 7, 8])
        r = p.predict(x_test[0:1], return_proba=True)
        self.assertIn(round(r[0]), [6, 7, 8])

        p.save("/tmp/test_predictor")
        p = ktrain.load_predictor("/tmp/test_predictor")
        r = p.predict(x_test[0:1])
        self.assertIn(round(r[0]), [6, 7, 8])


if __name__ == "__main__":
    main()
