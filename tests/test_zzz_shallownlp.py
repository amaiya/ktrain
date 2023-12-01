#!/usr/bin/env python3
"""
Tests of ktrain shallownlp module:
2020-05-26: renamed test_zzz_shallownlp.py because
            causes issues for tests following it when run in conjunction with test_regression.py.
"""
import os
from unittest import TestCase, main, skip

import numpy as np
import testenv

os.environ["DISABLE_V2_BEHAVIOR"] = "1"
from ktrain.text import shallownlp as snlp


class TestShallowNLP(TestCase):
    # @skip('temporarily disabled')
    def test_classifier(self):
        categories = [
            "alt.atheism",
            "soc.religion.christian",
            "comp.graphics",
            "sci.med",
        ]
        from sklearn.datasets import fetch_20newsgroups

        train_b = fetch_20newsgroups(
            subset="train", categories=categories, shuffle=True, random_state=42
        )
        test_b = fetch_20newsgroups(
            subset="test", categories=categories, shuffle=True, random_state=42
        )
        print("size of training set: %s" % (len(train_b["data"])))
        print("size of validation set: %s" % (len(test_b["data"])))
        print("classes: %s" % (train_b.target_names))
        x_train = train_b.data
        y_train = train_b.target
        x_test = test_b.data
        y_test = test_b.target
        classes = train_b.target_names

        clf = snlp.Classifier()
        clf.create_model("nbsvm", x_train, vec__ngram_range=(1, 3), vec__binary=True)
        clf.fit(x_train, y_train)
        self.assertGreaterEqual(clf.evaluate(x_test, y_test), 0.93)
        test_doc = "god christ jesus mother mary church sunday lord heaven amen"
        self.assertEqual(clf.predict(test_doc), 3)

    # @skip('temporarily disabled')
    def test_classifier_chinese(self):
        fpath = "./resources/text_data/chinese_hotel_reviews.csv"
        (x_train, y_train, label_names) = snlp.Classifier.load_texts_from_csv(
            fpath, text_column="content", label_column="pos", sep="|"
        )
        print("label names: %s" % (label_names))
        clf = snlp.Classifier()
        clf.create_model("nbsvm", x_train, vec__ngram_range=(1, 3), vec__binary=True)
        clf.fit(x_train, y_train)
        self.assertGreaterEqual(clf.evaluate(x_train, y_train), 0.98)
        neg_text = "我讨厌和鄙视这家酒店。"
        pos_text = "我喜欢这家酒店。"
        self.assertEqual(clf.predict(pos_text), 1)
        self.assertEqual(clf.predict(neg_text), 0)

    # @skip('temporarily disabled')
    def test_ner(self):
        ner = snlp.NER("en")
        text = """
        Xuetao Cao was head of the Chinese Academy of Medical Sciences and is
        the current president of Nankai University.
        """
        result = ner.predict(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][1], "PER")
        self.assertEqual(result[1][1], "ORG")
        self.assertEqual(result[2][1], "ORG")
        self.assertEqual(
            len(
                snlp.sent_tokenize(
                    "Paul Newman is a good actor.  Tommy Wisseau is not."
                )
            ),
            2,
        )

        ner = snlp.NER("zh")
        text = """
        曹雪涛曾任中国医学科学院院长，现任南开大学校长。
        """
        result = ner.predict(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][1], "PER")
        self.assertEqual(result[1][1], "ORG")
        self.assertEqual(result[2][1], "ORG")
        self.assertEqual(len(snlp.sent_tokenize("这是关于史密斯博士的第一句话。第二句话是关于琼斯先生的。")), 2)

        ner = snlp.NER("ru")
        text = """Владимир Владимирович Путин - российский политик, который является президентом России с 2012 года."""
        result = ner.predict(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], "PER")
        self.assertEqual(result[1][1], "LOC")

    # @skip('temporarily disabled')
    def test_search(self):
        document1 = """
        Hello there,

        Hope this email finds you well.

        Are you available to talk about our meeting?

        If so, let us plan to schedule the meeting
        at the Hefei National Laboratory for Physical Sciences at the Microscale.

        As I always say: живи сегодня надейся на завтра

        Sincerely,
        John Doe
        合肥微尺度国家物理科学实验室
        """

        document2 = """
        This is a random document with Arabic about our meeting.

        عش اليوم الأمل ليوم غد

        Bye for now.
        """

        docs = [document1, document2]

        result = snlp.search(
            ["physical sciences", "meeting", "Arabic"], docs, keys=["doc1", "doc2"]
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0][2], 1)
        self.assertEqual(result[1][2], 2)
        self.assertEqual(result[2][1], "meeting")
        self.assertEqual(result[3][1], "Arabic")

        result = snlp.search("合肥微尺度国家物理科学实验室", docs, keys=["doc1", "doc2"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], 7)

        result = snlp.search("сегодня надейся на завтра", docs, keys=["doc1", "doc2"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], 1)


if __name__ == "__main__":
    main()
