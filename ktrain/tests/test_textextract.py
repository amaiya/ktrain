#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
import testenv
import IPython
from unittest import TestCase, main, skip
import numpy as np
import ktrain
from ktrain import text 

class TestQA(TestCase):


    #@skip('temporarily disabled')
    def test_extract(self):
        path = './text_data/ktrain.pdf'
        from ktrain.text import TextExtractor
        te = TextExtractor()
        rawtext = te.extract(path)
        rawtext = rawtext.decode(errors='ignore')
        self.assertTrue(rawtext.startswith('ktrain'))

        #self.assertEqual(top_answer, 'in october of 1997')


if __name__ == "__main__":
    main()
