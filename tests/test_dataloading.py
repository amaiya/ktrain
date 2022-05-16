#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
import os.path
from unittest import TestCase, main, skip

import testenv

import ktrain
from ktrain import text as txt
from ktrain import utils as U
from ktrain import vision as vis

CURRDIR = os.path.dirname(__file__)


def texts_from_folder(preprocess_mode="standard"):
    DATADIR = os.path.join(CURRDIR, "resources/text_data/text_folder")
    trn, val, preproc = txt.texts_from_folder(
        DATADIR,
        max_features=100,
        maxlen=10,
        ngram_range=3,
        classes=["pos", "neg"],
        train_test_names=["train", "test"],
        preprocess_mode=preprocess_mode,
    )

    return (trn, val, preproc)


def texts_from_csv(preprocess_mode="standard"):
    DATA_PATH = os.path.join(CURRDIR, "resources/text_data/texts.csv")
    trn, val, preproc = txt.texts_from_csv(
        DATA_PATH,
        "text",
        val_filepath=DATA_PATH,
        label_columns=["neg", "pos"],
        max_features=100,
        maxlen=10,
        ngram_range=3,
        preprocess_mode=preprocess_mode,
    )
    return (trn, val, preproc)


def texts_from_csv_string(preprocess_mode="standard"):
    DATA_PATH = os.path.join(CURRDIR, "resources/text_data/texts-strings.csv")
    trn, val, preproc = txt.texts_from_csv(
        DATA_PATH,
        "text",
        val_filepath=DATA_PATH,
        label_columns="label",
        max_features=100,
        maxlen=10,
        ngram_range=3,
        preprocess_mode=preprocess_mode,
    )
    return (trn, val, preproc)


def texts_from_csv_int(preprocess_mode="standard"):
    DATA_PATH = os.path.join(CURRDIR, "resources/text_data/texts-ints.csv")
    trn, val, preproc = txt.texts_from_csv(
        DATA_PATH,
        "text",
        val_filepath=DATA_PATH,
        label_columns=["label"],
        max_features=100,
        maxlen=10,
        ngram_range=3,
        preprocess_mode=preprocess_mode,
    )
    return (trn, val, preproc)


def entities_from_conll2003():
    TDATA = os.path.join(CURRDIR, "resources/conll2003/train.txt")
    VDATA = os.path.join(CURRDIR, "resources/conll2003/valid.txt")
    (trn, val, preproc) = txt.entities_from_conll2003(TDATA, val_filepath=VDATA)
    return (trn, val, preproc)


def images_from_folder():
    (trn, val, preproc) = vis.images_from_folder(
        datadir=os.path.join(CURRDIR, "resources/image_data/image_folder"),
        data_aug=vis.get_data_aug(horizontal_flip=True),
        classes=["cat", "dog"],
        train_test_names=["train", "valid"],
    )
    return (trn, val, preproc)


def images_from_csv():
    train_fpath = os.path.join(CURRDIR, "resources/image_data/train-vision.csv")
    val_fpath = os.path.join(CURRDIR, "resources/image_data/valid-vision.csv")
    trn, val, preproc = vis.images_from_csv(
        train_fpath,
        "filename",
        directory=os.path.join(CURRDIR, "resources/image_data/image_folder/all"),
        val_filepath=val_fpath,
        label_columns=["cat", "dog"],
        data_aug=vis.get_data_aug(horizontal_flip=True),
    )
    return (trn, val, preproc)


def images_from_fname():
    trn, val, preproc = vis.images_from_fname(
        os.path.join(CURRDIR, "resources/image_data/image_folder/all"),
        pattern=r"([^/]+)\.\d+.jpg$",
        val_pct=0.25,
        random_state=42,
        data_aug=vis.get_data_aug(horizontal_flip=True),
    )
    return (trn, val, preproc)


def images_from_fname_regression():
    trn, val, preproc = vis.images_from_fname(
        os.path.join(CURRDIR, "resources/image_data/image_folder/all"),
        pattern=r"[^/]+\.(\d+).jpg$",
        val_pct=0.25,
        random_state=42,
        is_regression=True,
        data_aug=vis.get_data_aug(horizontal_flip=True),
    )
    return (trn, val, preproc)


class TestTextData(TestCase):
    def test_texts_from_folder_standard(self):
        (trn, val, preproc) = texts_from_folder()
        self.__test_texts_standard(trn, val, preproc)

    def test_texts_from_csv_standard(self):
        (trn, val, preproc) = texts_from_csv()
        self.__test_texts_standard(trn, val, preproc)

    def test_texts_from_csv_string_standard(self):
        (trn, val, preproc) = texts_from_csv_string()
        self.__test_texts_standard(trn, val, preproc)

    def test_texts_from_csv_int_standard(self):
        (trn, val, preproc) = texts_from_csv_int()
        self.__test_texts_standard(trn, val, preproc)

    def test_texts_from_folder_bert(self):
        (trn, val, preproc) = texts_from_folder(preprocess_mode="bert")
        self.__test_texts_bert(trn, val, preproc)

    def test_texts_from_csv_bert(self):
        (trn, val, preproc) = texts_from_csv(preprocess_mode="bert")
        self.__test_texts_bert(trn, val, preproc)

    def test_texts_from_csv_string_bert(self):
        (trn, val, preproc) = texts_from_csv_string(preprocess_mode="bert")
        self.__test_texts_bert(trn, val, preproc)

    def test_texts_from_csv_int_bert(self):
        (trn, val, preproc) = texts_from_csv_int(preprocess_mode="bert")
        self.__test_texts_bert(trn, val, preproc)

    def __test_texts_standard(self, trn, val, preproc):
        self.assertFalse(U.is_iter(trn))
        self.assertEqual(trn[0].shape, (4, 10))
        self.assertEqual(trn[1].shape, (4, 2))
        self.assertEqual(val[0].shape, (4, 10))
        self.assertEqual(val[1].shape, (4, 2))
        self.assertFalse(U.is_multilabel(trn))
        self.assertEqual(U.shape_from_data(trn), (4, 10))
        self.assertFalse(U.ondisk(trn))
        self.assertEqual(U.nsamples_from_data(trn), 4)
        self.assertEqual(U.nclasses_from_data(trn), 2)
        self.assertEqual(U.y_from_data(trn).shape, (4, 2))
        self.assertFalse(U.bert_data_tuple(trn))
        self.assertEqual(preproc.get_classes(), preproc.get_classes())
        self.assertEqual(preproc.ngram_count(), 3)
        self.assertEqual(preproc.preprocess(["hello book"])[0][-1], 1)
        self.assertEqual(preproc.preprocess(["hello book"]).shape, (1, 10))
        self.assertEqual(preproc.undo(val[0][0]), "the book is bad")

    def __test_texts_bert(self, trn, val, preproc):
        self.assertFalse(U.is_iter(trn))
        self.assertEqual(trn[0][0].shape, (4, 10))
        self.assertEqual(trn[1].shape, (4, 2))
        self.assertEqual(val[0][0].shape, (4, 10))
        self.assertEqual(val[1].shape, (4, 2))
        self.assertFalse(U.is_multilabel(trn))
        self.assertEqual(U.shape_from_data(trn), (4, 10))
        self.assertFalse(U.ondisk(trn))
        self.assertEqual(U.nsamples_from_data(trn), 4)
        self.assertEqual(U.nclasses_from_data(trn), 2)
        self.assertEqual(U.y_from_data(trn).shape, (4, 2))
        self.assertTrue(U.bert_data_tuple(trn))
        self.assertEqual(preproc.get_classes(), preproc.get_classes())
        self.assertEqual(preproc.preprocess(["hello book"])[0][0][0], 101)
        self.assertEqual(preproc.preprocess(["hello book"])[0].shape, (1, 10))
        self.assertEqual(preproc.undo(val[0][0][0]), "[CLS] the book is bad . [SEP]")


class TestNERData(TestCase):
    def test_entities_from_conll2003(self):
        (trn, val, preproc) = entities_from_conll2003()
        self.__test_ner(trn, val, preproc)

    def __test_ner(self, trn, val, preproc):
        self.assertTrue(U.is_iter(trn))
        self.assertTrue(U.is_ner(data=trn))
        self.assertFalse(U.is_multilabel(trn))
        self.assertEqual(U.shape_from_data(trn), (14041, 47))
        self.assertFalse(U.ondisk(trn))
        self.assertEqual(U.nsamples_from_data(trn), 14041)
        self.assertEqual(U.nclasses_from_data(trn), 10)
        self.assertEqual(len(U.y_from_data(trn)), 14041)
        self.assertFalse(U.bert_data_tuple(trn))
        self.assertEqual(
            preproc.get_classes(),
            [
                "<pad>",
                "O",
                "B-LOC",
                "B-PER",
                "B-ORG",
                "I-PER",
                "I-ORG",
                "B-MISC",
                "I-LOC",
                "I-MISC",
            ],
        )
        nerseq = preproc.preprocess(["hello world"])
        self.assertEqual(len(nerseq), 1)
        self.assertEqual(nerseq[0][0][0][0].tolist(), [21010, 100])


class TestImageData(TestCase):
    def test_images_from_folder(self):
        (trn, val, preproc) = images_from_folder()
        self.__test_images(trn, val, preproc)

    def test_images_from_csv(self):
        (trn, val, preproc) = images_from_csv()
        self.__test_images(trn, val, preproc)

    def test_images_from_fname(self):
        (trn, val, preproc) = images_from_fname()
        self.__test_images(trn, val, preproc, nsamples=18)

    def __test_images(self, trn, val, preproc, nsamples=16):
        self.assertTrue(U.is_iter(trn))
        self.assertEqual(U.shape_from_data(trn), (224, 224, 3))
        self.assertTrue(U.ondisk(trn))
        self.assertEqual(U.nsamples_from_data(trn), nsamples)
        self.assertEqual(U.nclasses_from_data(trn), 2)
        self.assertEqual(U.y_from_data(trn).shape, (nsamples, 2))
        self.assertFalse(U.bert_data_tuple(trn))
        self.assertEqual(preproc.get_classes(), ["cat", "dog"])
        (gen, steps) = preproc.preprocess(
            os.path.join(CURRDIR, "resources/image_data/image_folder/all")
        )
        self.assertEqual(type(gen).__name__, "DirectoryIterator")
        self.assertEqual(steps, 1)

    def test_images_from_fname_regression(self):
        (trn, val, preproc) = images_from_fname_regression()
        nsamples = 18
        self.assertTrue(U.is_iter(trn))
        self.assertEqual(U.shape_from_data(trn), (224, 224, 3))
        self.assertTrue(U.ondisk(trn))
        self.assertEqual(U.nsamples_from_data(trn), nsamples)
        # self.assertRaises(Exception, U.nclasses_from_data(trn))
        self.assertEqual(U.y_from_data(trn).shape, (nsamples,))
        self.assertFalse(U.bert_data_tuple(trn))
        self.assertEqual(preproc.get_classes(), [])
        (gen, steps) = preproc.preprocess(
            os.path.join(CURRDIR, "resources/image_data/image_folder/all")
        )
        self.assertEqual(type(gen).__name__, "DirectoryIterator")
        self.assertEqual(steps, 1)


if __name__ == "__main__":
    main()
