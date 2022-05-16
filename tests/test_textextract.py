#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
from unittest import TestCase, main, skip

import IPython
import numpy as np
import testenv

import ktrain
from ktrain import text


class TestTextExtraction(TestCase):

    # @skip('temporarily disabled')
    def test_tika_extract(self):
        path = "./resources/text_data/ktrain.pdf"
        from ktrain.text import TextExtractor

        te = TextExtractor(use_tika=True)
        rawtext = te.extract(path)
        self.assertTrue(rawtext.startswith("ktrain"))

    # @skip('temporarily disabled')
    def test_textract_extract(self):
        path = "./resources/text_data/ktrain.pdf"
        from ktrain.text import TextExtractor

        te = TextExtractor(use_tika=False)
        rawtext = te.extract(path)
        self.assertTrue(rawtext.startswith("ktrain"))


if __name__ == "__main__":
    main()
