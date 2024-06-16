#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
from unittest import TestCase, main, skip

import numpy as np
import testenv

from ktrain.text.kw import KeywordExtractor


class TestKeywordExtraction(TestCase):
    def test_ci(self):
        text = """Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias)."""
        kwe = KeywordExtractor()
        result = kwe.extract_keywords(text)
        print(result)
        self.assertEqual(result[0][0], "supervised learning")
        self.assertAlmostEqual(round(result[0][1], 2), 0.54)


if __name__ == "__main__":
    main()
