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
from ktrain.imports import ACC_NAME, VAL_ACC_NAME

class TestQA(TestCase):


    def test_qa(self):
        
        from sklearn.datasets import fetch_20newsgroups
        remove = ('headers', 'footers', 'quotes')
        newsgroups_train = fetch_20newsgroups(subset='train', remove=remove)
        newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)
        docs = newsgroups_train.data +  newsgroups_test.data

        tmp_folder = '/tmp/qa_test'
        text.SimpleQA.initialize_index(tmp_folder)
        text.SimpleQA.index_from_list(docs, tmp_folder, commit_every=len(docs))
        qa = text.SimpleQA(tmp_folder)

        answers = qa.ask('When did Cassini launch?')
        top_answer = answers[0]['answer']
        self.assertEqual(top_answer, 'in october of 1997')

if __name__ == "__main__":
    main()
