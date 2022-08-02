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
from ktrain.imports import ACC_NAME, VAL_ACC_NAME


class TestQA(TestCase):

    # @skip('temporarily disabled')
    def test_qa(self):

        from sklearn.datasets import fetch_20newsgroups

        remove = ("headers", "footers", "quotes")
        newsgroups_train = fetch_20newsgroups(subset="train", remove=remove)
        newsgroups_test = fetch_20newsgroups(subset="test", remove=remove)
        docs = newsgroups_train.data + newsgroups_test.data

        # tmp_folder = '/tmp/qa_test'
        import shutil
        import tempfile

        tmp_folder = tempfile.mkdtemp()
        shutil.rmtree(tmp_folder)
        text.SimpleQA.initialize_index(tmp_folder)
        text.SimpleQA.index_from_list(
            docs, tmp_folder, commit_every=len(docs), multisegment=True
        )
        qa = text.SimpleQA(tmp_folder, framework="tf")

        answers = qa.ask("When did Cassini launch?")
        top_answer = answers[0]["answer"]
        self.assertEqual(top_answer, "in october of 1997")

    @skip("temporarily disabled")
    def test_extractor(self):

        # data = ['Indeed, risk factors are sex, obesity, genetic factors and mechanical factors (3) .',
        #        'The sun is the center of our solar system.',
        #        'There is a risk of Donald Trump running again in 2024.',
        #        'My speciality is risk assessments.',
        #         """This risk was consistent across patients stratified by history of CVD, risk factors
        #         but no CVD, and neither CVD nor risk factors.""",
        #        """Risk factors associated with subsequent death include older age, hypertension, diabetes,
        #        ischemic heart disease, obesity and chronic lung disease; however, sometimes
        #         there are no obvious risk factors .""",
        #         'Three major risk factors for COVID-19 were sex (male), age (≥60), and severe pneumonia.']
        # from ktrain.text import AnswerExtractor
        # ae = AnswerExtractor()
        # import pandas as pd
        # pd.set_option("display.max_colwidth", None)
        # df = pd.DataFrame(data, columns=['Text'])
        # df = ae.extract(df.Text.values, df, [('What are the risk factors?', 'Risk Factors')], min_conf=8)
        # answers = df['Risk Factors'].values
        # self.assertEqual(answers[0].startswith('sex'), True)
        # self.assertEqual(answers[1], None)
        # self.assertEqual(answers[2], None)
        # self.assertEqual(answers[3], None)
        # self.assertEqual(answers[4], None)
        # self.assertEqual(answers[5].startswith('older'), True)
        # self.assertEqual(answers[6].startswith('sex'), True)

        data = [
            "Three major risk factors for COVID-19 were sex (male), age (≥60), and severe pneumonia.",
            "His speciality is medical risk assessments, and he is 30 years old.",
            "Results: A total of nine studies including 356 patients were included in this study, the mean age was 52.4 years and 221 (62.1%) were male.",
        ]
        from ktrain.text import AnswerExtractor

        ae = AnswerExtractor(framework="pt", device="cpu", quantize=True)
        import pandas as pd

        pd.set_option("display.max_colwidth", None)
        df = pd.DataFrame(data, columns=["Text"])
        import time

        start = time.time()
        df = ae.extract(
            df.Text.values,
            df,
            [
                ("What are the risk factors?", "Risk Factors"),
                ("How many individuals in sample?", "Sample Size"),
            ],
            min_conf=5,
        )
        print(time.time() - start)
        print(df.head())
        answers = df["Risk Factors"].values
        self.assertEqual(answers[0].startswith("sex"), True)
        self.assertEqual(answers[1], None)
        self.assertEqual(answers[2], None)
        answers = df["Sample Size"].values
        self.assertEqual(answers[0], None)
        self.assertEqual(answers[1], None)
        self.assertEqual(answers[2].startswith("356"), True)


if __name__ == "__main__":
    main()
