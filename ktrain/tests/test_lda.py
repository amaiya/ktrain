#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
from unittest import TestCase, main, skip

import IPython
import testenv

import ktrain
from ktrain.imports import ACC_NAME, VAL_ACC_NAME


class TestLDA(TestCase):
    def test_qa(self):
        rawtext = """
            Elon Musk leads Space Exploration Technologies (SpaceX), where he oversees
            the development and manufacturing of advanced rockets and spacecraft for missions
            to and beyond Earth orbit.
            """

        # collect data
        import numpy as np
        import pandas as pd
        from sklearn.datasets import fetch_20newsgroups

        remove = ("headers", "footers", "quotes")
        newsgroups_train = fetch_20newsgroups(subset="train", remove=remove)
        newsgroups_test = fetch_20newsgroups(subset="test", remove=remove)
        texts = newsgroups_train.data + newsgroups_test.data

        # buld and test LDA topic model
        tm = ktrain.text.get_topic_model(texts, n_features=10000)
        tm.build(texts, threshold=0.25)
        texts = tm.filter(texts)
        tags = tm.topics[np.argmax(tm.predict([rawtext]))]
        self.assertEqual(
            tags, "space nasa earth data launch surface solar moon mission planet"
        )
        tm.save("/tmp/tm")
        tm = ktrain.text.load_topic_model("/tmp/tm")
        tm.build(texts, threshold=0.25)
        tags = tm.topics[np.argmax(tm.predict([rawtext]))]
        self.assertEqual(
            tags, "space nasa earth data launch surface solar moon mission planet"
        )

        # document similarity
        tech_topics = [51, 85, 94, 22]
        tech_probs = tm.get_doctopics(topic_ids=tech_topics)
        doc_ids = [doc["doc_id"] for doc in tm.get_docs(topic_ids=tech_topics)]
        tm.train_scorer(topic_ids=tech_topics)
        other_topics = [i for i in range(tm.n_topics) if i not in tech_topics]
        other_texts = [d["text"] for d in tm.get_docs(topic_ids=other_topics)]
        other_scores = tm.score(other_texts)
        # display results in Pandas dataframe
        other_preds = [int(score > 0) for score in other_scores]
        data = sorted(
            list(zip(other_preds, other_scores, other_texts)),
            key=lambda item: item[1],
            reverse=True,
        )
        df = pd.DataFrame(data, columns=["Prediction", "Score", "Text"])
        self.assertTrue("recommendations for a laser printer" in df["Text"].values[0])

        # recommender
        tm.train_recommender()
        results = tm.recommend(text=rawtext, n=1)
        self.assertTrue(results[0]["text"].startswith("Archive-name"))


if __name__ == "__main__":
    main()
