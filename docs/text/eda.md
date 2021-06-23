Module ktrain.text.eda
======================

Classes
-------

`TopicModel(texts=None, n_topics=None, n_features=10000, min_df=5, max_df=0.5, stop_words='english', model_type='lda', lda_max_iter=5, lda_mode='online', token_pattern=None, verbose=1, hyperparam_kwargs=None)`
:   Fits a topic model to documents in <texts>.
    Example:
        tm = ktrain.text.get_topic_model(docs, n_topics=20, 
                                        n_features=1000, min_df=2, max_df=0.95)
    Args:
        texts (list of str): list of texts
        n_topics (int): number of topics.
                        If None, n_topics = min{400, sqrt[# documents/2]})
        n_features (int):  maximum words to consider
        max_df (float): words in more than max_df proportion of docs discarded
        stop_words (str or list): either 'english' for built-in stop words or
                                  a list of stop words to ignore
        model_type(str): type of topic model to fit. One of {'lda', 'nmf'}.  Default:'lda'
        lda_max_iter (int): maximum iterations for 'lda'.  5 is default if using lda_mode='online'.
                            If lda_mode='batch', this should be increased (e.g., 1500).
                            Ignored if model_type != 'lda'
        lda_mode (str):  one of {'online', 'batch'}. Ignored if model_type !='lda'
        token_pattern(str): regex pattern to use to tokenize documents. 
        verbose(bool): verbosity

    ### Instance variables

    `topics`
    :   convenience method/property

    ### Methods

    `build(self, texts, threshold=None)`
    :   Builds the document-topic distribution showing the topic probability distirbution
        for each document in <texts> with respect to the learned topic space.
        Args:
            texts (list of str): list of text documents
            threshold (float): If not None, documents with whose highest topic probability
                               is less than threshold are filtered out.

    `filter(self, lst)`
    :   The build method may prune documents based on threshold.
        This method prunes other lists based on how build pruned documents.
        This is useful to filter lists containing metadata associated with documents
        for use with visualize_documents.
        Args:
            lst(list): a list of data
        Returns:
            list:  a filtered list of data based on how build filtered the documents

    `get_docs(self, topic_ids=[], doc_ids=[], rank=False)`
    :   Returns document entries for supplied topic_ids.
        Documents returned are those whose primary topic is topic with given topic_id
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
            rank(bool): If True, the list is sorted first by topic_id (ascending)
                        and then ty topic probability (descending).
                        Otherwise, list is sorted by doc_id (i.e., the order
                        of texts supplied to self.build (which is the order of self.doc_topics).
        
        Returns:
            list of dicts:  list of dicts with keys:
                            'text': text of document
                            'doc_id': ID of document
                            'topic_proba': topic probability (or score)
                            'topic_id': ID of topic

    `get_doctopics(self, topic_ids=[], doc_ids=[])`
    :   Returns a topic probability distribution for documents
        with primary topic that is one of <topic_ids> and with doc_id in <doc_ids>.
        
        If no topic_ids or doc_ids are provided, then topic distributions for all documents
        are returned (which equivalent to the output of get_document_topic_distribution).
        
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
        Returns:
            np.ndarray: Each row is the topic probability distribution of a document.
                        Array is sorted in the order returned by self.get_docs.

    `get_document_topic_distribution(self)`
    :   Gets the document-topic distribution.
        Each row is a document and each column is a topic
        The output of this method is equivalent to invoking get_doctopics with no arguments.

    `get_sorted_docs(self, topic_id)`
    :   Returns all docs sorted by relevance to <topic_id>.
        Unlike get_docs, this ranks documents by the supplied topic_id rather
        than the topic_id to which document is most relevant.

    `get_texts(self, topic_ids=[])`
    :   Returns texts for documents
        with primary topic that is one of <topic_ids>
        Args:
            topic_ids(list of ints): list of topic IDs
        Returns:
            list of str

    `get_topics(self, n_words=10, as_string=True)`
    :   Returns a list of discovered topics
        Args:
            n_words(int): number of words to use in topic summary
            as_string(bool): If True, each summary is a space-delimited string instead of list of words

    `get_word_weights(self, topic_id, n_words=100)`
    :   Returns a list tuples of the form: (word, weight) for given topic_id.
        The weight can be interpreted as the number of times word was assigned to topic with given topic_id.
        REFERENCE: https://stackoverflow.com/a/48890889/13550699
        Args:
            topic_id(int): topic ID
            n_words=int): number of top words

    `predict(self, texts, threshold=None, harden=False)`
    :   Args:
            texts (list of str): list of texts
            threshold (float): If not None, documents with maximum topic scores
                                less than <threshold> are filtered out
            harden(bool): If True, each document is assigned to a single topic for which
                          it has the highest score
        Returns:
            if threshold is None:
                np.ndarray: topic distribution for each text document
            else:
                (np.ndarray, np.ndarray): topic distribution and boolean array

    `print_topics(self, n_words=10, show_counts=False)`
    :   print topics
        n_words(int): number of words to describe each topic
        show_counts(bool): If True, print topics with document counts, where
                           the count is the number of documents with that topic as primary.

    `recommend(self, text=None, doc_topic=None, n=5, n_neighbors=100)`
    :   Given an example document, recommends documents similar to it
        from the set of documents supplied to build().
        
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
            n (int): number of recommendations to return
        Returns:
            list of tuples: each tuple is of the form:
                            (text, doc_id, topic_probability, topic_id)

    `save(self, fname)`
    :   save TopicModel object

    `score(self, texts=None, doc_topics=None)`
    :   Given a new set of documents (supplied as texts or doc_topics), the score method
        uses a One-Class classifier to score documents based on similarity to a
        seed set of documents (where seed set is computed by train_scorer() method).
        
        Higher scores indicate a higher degree of similarity.
        Positive values represent a binary decision of similar.
        Negative values represent a binary decision of dissimlar.
        In practice, negative scores closer to zer will also be simlar as One-Class
        classifiers are more strict than traditional binary classifiers.
        Documents with negative scores closer to zero are good candidates for
        inclusion in a training set for binary classification (e.g., active labeling).
        
        NOTE: The score method currently employs the use of LocalOutLierFactor, which
        means you should not try to score documents that were used in training. Only
        new, unseen documents should be scored for similarity.
        
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
        Returns:
            list of floats:  larger values indicate higher degree of similarity
                             positive values indicate a binary decision of similar
                             negative values indicate binary decision of dissimilar
                             In practice, negative scores closer to zero will also 
                             be similar as One-class classifiers are more strict
                             than traditional binary classifiers.

    `search(self, query, topic_ids=[], doc_ids=[], case_sensitive=False)`
    :   search documents for query string.
        Args:
            query(str):  the word or phrase to search
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
            case_sensitive(bool):  If True, case sensitive search

    `train(self, texts, model_type='lda', n_topics=None, n_features=10000, min_df=5, max_df=0.5, stop_words='english', lda_max_iter=5, lda_mode='online', token_pattern=None, hyperparam_kwargs=None)`
    :   Fits a topic model to documents in <texts>.
        Example:
            tm = ktrain.text.get_topic_model(docs, n_topics=20, 
                                            n_features=1000, min_df=2, max_df=0.95)
        Args:
            texts (list of str): list of texts
            n_topics (int): number of topics.
                            If None, n_topics = min{400, sqrt[# documents/2]})
            n_features (int):  maximum words to consider
            max_df (float): words in more than max_df proportion of docs discarded
            stop_words (str or list): either 'english' for built-in stop words or
                                      a list of stop words to ignore
            lda_max_iter (int): maximum iterations for 'lda'.  5 is default if using lda_mode='online'.
                                If lda_mode='batch', this should be increased (e.g., 1500).
                                Ignored if model_type != 'lda'
            lda_mode (str):  one of {'online', 'batch'}. Ignored of model_type !='lda'
            token_pattern(str): regex pattern to use to tokenize documents. 
                                If None, a default tokenizer will be used
            hyperparam_kwargs(dict): hyperparameters for LDA/NMF
                                     Keys in this dict can be any of the following:
                                         alpha: alpha for LDA  default: 5./n_topics
                                         beta: beta for LDA.  default:0.01
                                         nmf_alpha: alpha for NMF.  default:0
                                         l1_ratio: l1_ratio for NMF. default: 0
                                         ngram_range:  whether to consider bigrams, trigrams. default: (1,1) 
                                    
        Returns:
            tuple: (model, vectorizer)

    `train_recommender(self, n_neighbors=20, metric='minkowski', p=2)`
    :   Trains a recommender that, given a single document, will return
        documents in the corpus that are semantically similar to it.
        
        Args:
            n_neighbors (int): 
        Returns:
            None

    `train_scorer(self, topic_ids=[], doc_ids=[], n_neighbors=20)`
    :   Trains a scorer that can score documents based on similarity to a
        seed set of documents represented by topic_ids and doc_ids.
        
        NOTE: The score method currently employs the use of LocalOutLierFactor, which
        means you should not try to score documents that were used in training. Only
        new, unseen documents should be scored for similarity. 
        REFERENCE: 
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
        
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).  Documents associated
                                     with these topic_ids will be used as seed set.
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics.  Documents associated 
                                    with these doc_ids will be used as seed set.
        Returns:
            None

    `visualize_documents(self, texts=None, doc_topics=None, width=700, height=700, point_size=5, title='Document Visualization', extra_info={}, colors=None, filepath=None)`
    :   Generates a visualization of a set of documents based on model.
        If <texts> is supplied, raw documents will be first transformed into document-topic
        matrix.  If <doc_topics> is supplied, then this will be used for visualization instead.
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
            width(int): width of image
            height(int): height of image
            point_size(int): size of circles in plot
            title(str):  title of visualization
            extra_info(dict of lists): A user-supplied information for each datapoint (attributes of the datapoint).
                                       The keys are field names.  The values are lists - each of which must
                                       be the same number of elements as <texts> or <doc_topics>. These fields are displayed
                                       when hovering over datapoints in the visualization.
            colors(list of str):  list of Hex color codes for each datapoint.
                                  Length of list must match either len(texts) or doc_topics.shape[0]
            filepath(str):             Optional filepath to save the interactive visualization

`get_topic_model(texts=None, n_topics=None, n_features=10000, min_df=5, max_df=0.5, stop_words='english', model_type='lda', lda_max_iter=5, lda_mode='online', token_pattern=None, verbose=1, hyperparam_kwargs=None)`
:   Fits a topic model to documents in <texts>.
    Example:
        tm = ktrain.text.get_topic_model(docs, n_topics=20, 
                                        n_features=1000, min_df=2, max_df=0.95)
    Args:
        texts (list of str): list of texts
        n_topics (int): number of topics.
                        If None, n_topics = min{400, sqrt[# documents/2]})
        n_features (int):  maximum words to consider
        max_df (float): words in more than max_df proportion of docs discarded
        stop_words (str or list): either 'english' for built-in stop words or
                                  a list of stop words to ignore
        model_type(str): type of topic model to fit. One of {'lda', 'nmf'}.  Default:'lda'
        lda_max_iter (int): maximum iterations for 'lda'.  5 is default if using lda_mode='online'.
                            If lda_mode='batch', this should be increased (e.g., 1500).
                            Ignored if model_type != 'lda'
        lda_mode (str):  one of {'online', 'batch'}. Ignored if model_type !='lda'
        token_pattern(str): regex pattern to use to tokenize documents. 
        verbose(bool): verbosity

    ### Instance variables

    `topics`
    :   convenience method/property

    ### Methods

    `build(self, texts, threshold=None)`
    :   Builds the document-topic distribution showing the topic probability distirbution
        for each document in <texts> with respect to the learned topic space.
        Args:
            texts (list of str): list of text documents
            threshold (float): If not None, documents with whose highest topic probability
                               is less than threshold are filtered out.

    `filter(self, lst)`
    :   The build method may prune documents based on threshold.
        This method prunes other lists based on how build pruned documents.
        This is useful to filter lists containing metadata associated with documents
        for use with visualize_documents.
        Args:
            lst(list): a list of data
        Returns:
            list:  a filtered list of data based on how build filtered the documents

    `get_docs(self, topic_ids=[], doc_ids=[], rank=False)`
    :   Returns document entries for supplied topic_ids.
        Documents returned are those whose primary topic is topic with given topic_id
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
            rank(bool): If True, the list is sorted first by topic_id (ascending)
                        and then ty topic probability (descending).
                        Otherwise, list is sorted by doc_id (i.e., the order
                        of texts supplied to self.build (which is the order of self.doc_topics).
        
        Returns:
            list of dicts:  list of dicts with keys:
                            'text': text of document
                            'doc_id': ID of document
                            'topic_proba': topic probability (or score)
                            'topic_id': ID of topic

    `get_doctopics(self, topic_ids=[], doc_ids=[])`
    :   Returns a topic probability distribution for documents
        with primary topic that is one of <topic_ids> and with doc_id in <doc_ids>.
        
        If no topic_ids or doc_ids are provided, then topic distributions for all documents
        are returned (which equivalent to the output of get_document_topic_distribution).
        
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
        Returns:
            np.ndarray: Each row is the topic probability distribution of a document.
                        Array is sorted in the order returned by self.get_docs.

    `get_document_topic_distribution(self)`
    :   Gets the document-topic distribution.
        Each row is a document and each column is a topic
        The output of this method is equivalent to invoking get_doctopics with no arguments.

    `get_sorted_docs(self, topic_id)`
    :   Returns all docs sorted by relevance to <topic_id>.
        Unlike get_docs, this ranks documents by the supplied topic_id rather
        than the topic_id to which document is most relevant.

    `get_texts(self, topic_ids=[])`
    :   Returns texts for documents
        with primary topic that is one of <topic_ids>
        Args:
            topic_ids(list of ints): list of topic IDs
        Returns:
            list of str

    `get_topics(self, n_words=10, as_string=True)`
    :   Returns a list of discovered topics
        Args:
            n_words(int): number of words to use in topic summary
            as_string(bool): If True, each summary is a space-delimited string instead of list of words

    `get_word_weights(self, topic_id, n_words=100)`
    :   Returns a list tuples of the form: (word, weight) for given topic_id.
        The weight can be interpreted as the number of times word was assigned to topic with given topic_id.
        REFERENCE: https://stackoverflow.com/a/48890889/13550699
        Args:
            topic_id(int): topic ID
            n_words=int): number of top words

    `predict(self, texts, threshold=None, harden=False)`
    :   Args:
            texts (list of str): list of texts
            threshold (float): If not None, documents with maximum topic scores
                                less than <threshold> are filtered out
            harden(bool): If True, each document is assigned to a single topic for which
                          it has the highest score
        Returns:
            if threshold is None:
                np.ndarray: topic distribution for each text document
            else:
                (np.ndarray, np.ndarray): topic distribution and boolean array

    `print_topics(self, n_words=10, show_counts=False)`
    :   print topics
        n_words(int): number of words to describe each topic
        show_counts(bool): If True, print topics with document counts, where
                           the count is the number of documents with that topic as primary.

    `recommend(self, text=None, doc_topic=None, n=5, n_neighbors=100)`
    :   Given an example document, recommends documents similar to it
        from the set of documents supplied to build().
        
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
            n (int): number of recommendations to return
        Returns:
            list of tuples: each tuple is of the form:
                            (text, doc_id, topic_probability, topic_id)

    `save(self, fname)`
    :   save TopicModel object

    `score(self, texts=None, doc_topics=None)`
    :   Given a new set of documents (supplied as texts or doc_topics), the score method
        uses a One-Class classifier to score documents based on similarity to a
        seed set of documents (where seed set is computed by train_scorer() method).
        
        Higher scores indicate a higher degree of similarity.
        Positive values represent a binary decision of similar.
        Negative values represent a binary decision of dissimlar.
        In practice, negative scores closer to zer will also be simlar as One-Class
        classifiers are more strict than traditional binary classifiers.
        Documents with negative scores closer to zero are good candidates for
        inclusion in a training set for binary classification (e.g., active labeling).
        
        NOTE: The score method currently employs the use of LocalOutLierFactor, which
        means you should not try to score documents that were used in training. Only
        new, unseen documents should be scored for similarity.
        
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
        Returns:
            list of floats:  larger values indicate higher degree of similarity
                             positive values indicate a binary decision of similar
                             negative values indicate binary decision of dissimilar
                             In practice, negative scores closer to zero will also 
                             be similar as One-class classifiers are more strict
                             than traditional binary classifiers.

    `search(self, query, topic_ids=[], doc_ids=[], case_sensitive=False)`
    :   search documents for query string.
        Args:
            query(str):  the word or phrase to search
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
            case_sensitive(bool):  If True, case sensitive search

    `train(self, texts, model_type='lda', n_topics=None, n_features=10000, min_df=5, max_df=0.5, stop_words='english', lda_max_iter=5, lda_mode='online', token_pattern=None, hyperparam_kwargs=None)`
    :   Fits a topic model to documents in <texts>.
        Example:
            tm = ktrain.text.get_topic_model(docs, n_topics=20, 
                                            n_features=1000, min_df=2, max_df=0.95)
        Args:
            texts (list of str): list of texts
            n_topics (int): number of topics.
                            If None, n_topics = min{400, sqrt[# documents/2]})
            n_features (int):  maximum words to consider
            max_df (float): words in more than max_df proportion of docs discarded
            stop_words (str or list): either 'english' for built-in stop words or
                                      a list of stop words to ignore
            lda_max_iter (int): maximum iterations for 'lda'.  5 is default if using lda_mode='online'.
                                If lda_mode='batch', this should be increased (e.g., 1500).
                                Ignored if model_type != 'lda'
            lda_mode (str):  one of {'online', 'batch'}. Ignored of model_type !='lda'
            token_pattern(str): regex pattern to use to tokenize documents. 
                                If None, a default tokenizer will be used
            hyperparam_kwargs(dict): hyperparameters for LDA/NMF
                                     Keys in this dict can be any of the following:
                                         alpha: alpha for LDA  default: 5./n_topics
                                         beta: beta for LDA.  default:0.01
                                         nmf_alpha: alpha for NMF.  default:0
                                         l1_ratio: l1_ratio for NMF. default: 0
                                         ngram_range:  whether to consider bigrams, trigrams. default: (1,1) 
                                    
        Returns:
            tuple: (model, vectorizer)

    `train_recommender(self, n_neighbors=20, metric='minkowski', p=2)`
    :   Trains a recommender that, given a single document, will return
        documents in the corpus that are semantically similar to it.
        
        Args:
            n_neighbors (int): 
        Returns:
            None

    `train_scorer(self, topic_ids=[], doc_ids=[], n_neighbors=20)`
    :   Trains a scorer that can score documents based on similarity to a
        seed set of documents represented by topic_ids and doc_ids.
        
        NOTE: The score method currently employs the use of LocalOutLierFactor, which
        means you should not try to score documents that were used in training. Only
        new, unseen documents should be scored for similarity. 
        REFERENCE: 
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
        
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).  Documents associated
                                     with these topic_ids will be used as seed set.
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics.  Documents associated 
                                    with these doc_ids will be used as seed set.
        Returns:
            None

    `visualize_documents(self, texts=None, doc_topics=None, width=700, height=700, point_size=5, title='Document Visualization', extra_info={}, colors=None, filepath=None)`
    :   Generates a visualization of a set of documents based on model.
        If <texts> is supplied, raw documents will be first transformed into document-topic
        matrix.  If <doc_topics> is supplied, then this will be used for visualization instead.
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
            width(int): width of image
            height(int): height of image
            point_size(int): size of circles in plot
            title(str):  title of visualization
            extra_info(dict of lists): A user-supplied information for each datapoint (attributes of the datapoint).
                                       The keys are field names.  The values are lists - each of which must
                                       be the same number of elements as <texts> or <doc_topics>. These fields are displayed
                                       when hovering over datapoints in the visualization.
            colors(list of str):  list of Hex color codes for each datapoint.
                                  Length of list must match either len(texts) or doc_topics.shape[0]
            filepath(str):             Optional filepath to save the interactive visualization