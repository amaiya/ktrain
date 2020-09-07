from ..imports import *
from .. import utils as U
from . import textutils as TU
from . import preprocessor as pp
import time

class TopicModel():


    def __init__(self,texts=None, n_topics=None, n_features=10000, 
                 min_df=5, max_df=0.5,  stop_words='english',
                 model_type='lda',
                 lda_max_iter=5, lda_mode='online',
                 token_pattern=None, verbose=1,
                 hyperparam_kwargs=None
    ):
        """
        Fits a topic model to documents in <texts>.
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
            verbose(bool): verbosity

        """
        self.verbose=verbose

        # estimate n_topics
        if n_topics is None:
            if texts is None:
                raise ValueError('If n_topics is None, texts must be supplied')
            estimated = max(1, int(math.floor(math.sqrt(len(texts) / 2))))
            n_topics = min(400, estimated)
            print('n_topics automatically set to %s' % (n_topics))

        # train model
        if texts is not None:
            (model, vectorizer) = self.train(texts, model_type=model_type,
                                             n_topics=n_topics, n_features=n_features,
                                             min_df = min_df, max_df = max_df, 
                                             stop_words=stop_words,
                                             lda_max_iter=lda_max_iter, lda_mode=lda_mode,
                                             token_pattern=token_pattern,
                                             hyperparam_kwargs=hyperparam_kwargs)
        else:
            vectorizer = None
            model = None



        # save model and vectorizer and hyperparameter settings
        self.vectorizer = vectorizer
        self.model = model
        self.n_topics = n_topics
        self.n_features = n_features
        if verbose: print('done.')

        # these variables are set by self.build():
        self.topic_dict = None
        self.doc_topics = None
        self.bool_array = None

        self.scorer = None       # set by self.train_scorer()
        self.recommender = None  # set by self.train_recommender()
        return


    def train(self,texts, model_type='lda', n_topics=None, n_features=10000,
              min_df=5, max_df=0.5,  stop_words='english',
              lda_max_iter=5, lda_mode='online',
              token_pattern=None, hyperparam_kwargs=None):
        """
        Fits a topic model to documents in <texts>.
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
        """
        if hyperparam_kwargs is None:
            hyperparam_kwargs = {}
        alpha = hyperparam_kwargs.get('alpha', 5.0 / n_topics)
        beta = hyperparam_kwargs.get('beta', 0.01)
        nmf_alpha = hyperparam_kwargs.get('nmf_alpha', 0)
        l1_ratio = hyperparam_kwargs.get('l1_ratio', 0)
        ngram_range = hyperparam_kwargs.get('ngram_range', (1,1))

        # adjust defaults based on language detected
        if texts is not None:
            lang = TU.detect_lang(texts)
            if lang != 'en':
                stopwords = None if stop_words=='english' else stop_words
                token_pattern = r'(?u)\b\w+\b' if token_pattern is None else token_pattern
            if pp.is_nospace_lang(lang):
                text_list = []
                for t in texts:
                    text_list.append(' '.join(jieba.cut(t, HMM=False)))
                texts = text_list
            if self.verbose: print('lang: %s' % (lang))


        # preprocess texts
        if self.verbose: print('preprocessing texts...')
        if token_pattern is None: token_pattern = TU.DEFAULT_TOKEN_PATTERN
        #if token_pattern is None: token_pattern = r'(?u)\b\w\w+\b'
        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df,
                                 max_features=n_features, stop_words=stop_words,
                                 token_pattern=token_pattern, ngram_range=ngram_range)
        

        x_train = vectorizer.fit_transform(texts)

        # fit model

        if self.verbose: print('fitting model...')
        if model_type == 'lda':
            model = LatentDirichletAllocation(n_components=n_topics, max_iter=lda_max_iter,
                                              learning_method=lda_mode, learning_offset=50.,
                                              doc_topic_prior=alpha,
                                              topic_word_prior=beta,
                                              verbose=self.verbose, random_state=0)
        elif model_type == 'nmf':
            model = NMF(
                n_components=n_topics,
                max_iter=lda_max_iter,
                verbose=self.verbose,
                alpha=nmf_alpha,
                l1_ratio=l1_ratio,
                random_state=0)
        else:
            raise ValueError("unknown model type:", str(model_type))
        model.fit(x_train)

        # save model and vectorizer and hyperparameter settings
        return (model, vectorizer)


    @property
    def topics(self):
        """
        convenience method/property
        """
        return self.get_topics()


    def get_word_weights(self, topic_id, n_words=100):
        """
        Returns a list tuples of the form: (word, weight) for given topic_id.
        The weight can be interpreted as the number of times word was assigned to topic with given topic_id.
        REFERENCE: https://stackoverflow.com/a/48890889/13550699
        Args:
            topic_id(int): topic ID
            n_words=int): number of top words
        """
        self._check_model()
        if topic_id+1 > len(self.model.components_): 
            raise ValueError('topic_id must be less than %s' % (len(self.model.components_)))
        feature_names = self.vectorizer.get_feature_names()
        word_probs = self.model.components_[topic_id]
        word_ids = [i for i in word_probs.argsort()[:-n_words - 1:-1]]
        words = [feature_names[i] for i in word_ids]
        probs = [word_probs[i] for i in word_ids]
        return list( zip(words, probs) )


    def get_topics(self, n_words=10, as_string=True):
        """
        Returns a list of discovered topics
        Args:
            n_words(int): number of words to use in topic summary
            as_string(bool): If True, each summary is a space-delimited string instead of list of words
        """
        self._check_model()
        feature_names = self.vectorizer.get_feature_names()
        topic_summaries = []
        for topic_idx, topic in enumerate(self.model.components_):
            summary = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            if as_string: summary = " ".join(summary)
            topic_summaries.append(summary)
        return topic_summaries


    def print_topics(self, n_words=10, show_counts=False):
        """
        print topics
        """
        topics = self.get_topics(n_words=n_words, as_string=True)
        if show_counts:
            self._check_build()
            topic_counts = sorted([ (k, topics[k], len(v)) for k,v in self.topic_dict.items()], 
                                    key=lambda kv:kv[-1], reverse=True)
            for (idx, topic, count) in topic_counts:
                print("topic:%s | count:%s | %s" %(idx, count, topic))
        else:
            for i, t in enumerate(topics):
                print('topic %s | %s' % (i, t))
        return


    def build(self, texts, threshold=None):
        """
        Builds the document-topic distribution showing the topic probability distirbution
        for each document in <texts> with respect to the learned topic space.
        Args:
            texts (list of str): list of text documents
            threshold (float): If not None, documents with whose highest topic probability
                               is less than threshold are filtered out.
        """
        doc_topics, bool_array = self.predict(texts, threshold=threshold)
        self.doc_topics = doc_topics
        self.bool_array = bool_array

        texts = [text for i, text in enumerate(texts) if bool_array[i]]
        self.topic_dict = self._rank_documents(texts, doc_topics=doc_topics)
        return


    def filter(self, lst):
        """
        The build method may prune documents based on threshold.
        This method prunes other lists based on how build pruned documents.
        This is useful to filter lists containing metadata associated with documents
        for use with visualize_documents.
        Args:
            lst(list): a list of data
        Returns:
            list:  a filtered list of data based on how build filtered the documents
        """
        if len(lst) != self.bool_array.shape[0]:
            raise ValueError('Length of lst is not consistent with the number of documents ' +
                             'supplied to get_topic_model')
        arr = np.array(lst)
        return list(arr[self.bool_array])
                           

    
    def get_docs(self, topic_ids=[], doc_ids=[], rank=False):
        """
        Returns document entries for supplied topic_ids
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
            list of tuples:  list of tuples where each tuple is of the form 
                             (text, doc_id, topic_probability, topic_id).
            
        """
        self._check_build()
        if not topic_ids:
            topic_ids = list(range(self.n_topics))
        result_texts = []
        for topic_id in topic_ids:
            if topic_id not in self.topic_dict: continue
            texts = [tup + (topic_id,) for tup in self.topic_dict[topic_id] 
                                           if not doc_ids or tup[1] in doc_ids]
            result_texts.extend(texts)
        if not rank:
            result_texts = sorted(result_texts, key=lambda x:x[1])
        return result_texts


    def get_doctopics(self,  topic_ids=[], doc_ids=[]):
        """
        Returns a topic probability distribution for documents
        with primary topic that is one of <topic_ids>
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
        Returns:
            np.ndarray: Each row is the topic probability distribution of a document.
                        Array is sorted in the order returned by self.get_docs.
                        
        """
        docs = self.get_docs(topic_ids=topic_ids, doc_ids=doc_ids)
        return np.array([self.doc_topics[idx] for idx in [x[1] for x in docs]])


    def get_texts(self,  topic_ids=[]):
        """
        Returns texts for documents
        with primary topic that is one of <topic_ids>
        Args:
            topic_ids(list of ints): list of topic IDs
        Returns:
            list of str
        """
        if not topic_ids: topic_ids = list(range(self.n_topics))
        docs = self.get_docs(topic_ids)
        return [x[0] for x in docs]



    def predict(self, texts, threshold=None, harden=False):
        """
        Args:
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
        """
        self._check_model()
        transformed_texts = self.vectorizer.transform(texts)
        X_topics = self.model.transform(transformed_texts)
        #if self.model_type == 'nmf':
            #scores = np.matrix(X_topics)
            #scores_normalized= scores/scores.sum(axis=1)
            #X_topics = scores_normalized
        _idx = np.array([True] * len(texts))
        if threshold is not None:
            _idx = np.amax(X_topics, axis=1) > threshold  # idx of doc that above the threshold
            _idx = np.array(_idx)
            X_topics = X_topics[_idx]
        if harden: X_topics = self._harden_topics(X_topics)
        if threshold is not None:
            return (X_topics, _idx)
        else:
            return X_topics


    def visualize_documents(self, texts=None, doc_topics=None, 
                            width=700, height=700, point_size=5, title='Document Visualization',
                            extra_info={},
                            colors=None,
                            filepath=None,):
        """
        Generates a visualization of a set of documents based on model.
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
        """

        # error-checking
        if texts is not None: length = len(texts)
        else: length = doc_topics.shape[0]
        if colors is not None and len(colors) != length:
            raise ValueError('length of colors is not consistent with length of texts or doctopics')
        if texts is not None and doc_topics is not None:
            raise ValueError('texts is mutually-exclusive with doc_topics')
        if texts is None and doc_topics is None:
            raise ValueError('One of texts or doc_topics is required.')
        if extra_info:
            invalid_keys = ['x', 'y', 'topic', 'fill_color']
            for k in extra_info.keys():
                if k in invalid_keys:
                    raise ValueError('cannot use "%s" as key in extra_info' %(k))
                lst = extra_info[k]
                if len(lst) != length:
                    raise ValueError('texts and extra_info lists must be same size')

        # check fo bokeh
        try:
            import bokeh.plotting as bp
            from bokeh.plotting import save
            from bokeh.models import HoverTool
            from bokeh.io import output_notebook
        except:
            warnings.warn('visualize_documents method requires bokeh package: pip install bokeh')
            return

        # prepare data
        if doc_topics is not None:
            X_topics = doc_topics
        else:
            if self.verbose:  print('transforming texts...', end='')
            X_topics = self.predict(texts, harden=False)
            if self.verbose: print('done.')

        # reduce to 2-D
        if self.verbose:  print('reducing to 2 dimensions...', end='')
        tsne_model = TSNE(n_components=2, verbose=self.verbose, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(X_topics)
        print('done.')

        # get random colormap
        colormap = U.get_random_colors(self.n_topics)

        # generate inline visualization in Jupyter notebook
        lda_keys = self._harden_topics(X_topics)
        if colors is None: colors = colormap[lda_keys]
        topic_summaries = self.get_topics(n_words=5)
        os.environ["BOKEH_RESOURCES"]="inline"
        output_notebook()
        dct = { 
                'x':tsne_lda[:,0],
                'y':tsne_lda[:, 1],
                'topic':[topic_summaries[tid] for tid in lda_keys],
                'fill_color':colors,}
        tool_tups = [('index', '$index'),
                     ('(x,y)','($x,$y)'),
                     ('topic', '@topic')]
        for k in extra_info.keys():
            dct[k] = extra_info[k]
            tool_tups.append((k, '@'+k))

        source = bp.ColumnDataSource(data=dct)
        hover = HoverTool( tooltips=tool_tups)
        p = bp.figure(plot_width=width, plot_height=height, 
                      tools=[hover, 'save', 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
                      #tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                      title=title)
        #plot_lda = bp.figure(plot_width=1400, plot_height=1100,
			   #title=title,
			   #tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
			   #x_axis_type=None, y_axis_type=None, min_border=1)
        p.circle('x', 'y', size=point_size, source=source, fill_color= 'fill_color')
        bp.show(p)
        if filepath is not None:
            bp.output_file(filepath)
            bp.save(p)
        return


    def train_recommender(self, n_neighbors=20, metric='minkowski', p=2):
        """
        Trains a recommender that, given a single document, will return
        documents in the corpus that are semantically similar to it.

        Args:
            n_neighbors (int): 
        Returns:
            None
        """
        from sklearn.neighbors import NearestNeighbors
        rec = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, p=p)
        probs = self.get_doctopics()
        rec.fit(probs)
        self.recommender = rec
        return



    def recommend(self, text=None, doc_topic=None, n=5, n_neighbors=100):
        """
        Given an example document, recommends documents similar to it
        from the set of documents supplied to build().
 
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
            n (int): number of recommendations to return
        Returns:
            list of tuples: each tuple is of the form:
                            (text, doc_id, topic_probability, topic_id)

        """
        # error-checks
        if text is not None and doc_topic is not None:
            raise ValueError('text is mutually-exclusive with doc_topic')
        if text is None and doc_topic is None:
            raise ValueError('One of text or doc_topic is required.')
        if text is not None and type(text) not in [str]:
            raise ValueError('text must be a str ')
        if  doc_topic is not None and type(doc_topic) not in [np.ndarray]:
            raise ValueError('doc_topic must be a np.ndarray')

        if n > n_neighbors: n_neighbors = n

        x_test = [doc_topic]
        if text:
            x_test = self.predict([text])
        docs = self.get_docs()
        indices = self.recommender.kneighbors(x_test, return_distance=False, n_neighbors=n_neighbors)
        results = [doc for i, doc in enumerate(docs) if i in indices]
        return results[:n]




    def train_scorer(self, topic_ids=[], doc_ids=[], n_neighbors=20):
        """
        Trains a scorer that can score documents based on similarity to a
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
        """
        from sklearn.neighbors import LocalOutlierFactor
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=0.1)
        probs = self.get_doctopics(topic_ids=topic_ids, doc_ids=doc_ids)
        clf.fit(probs)
        self.scorer = clf
        return



    def score(self, texts=None, doc_topics=None):
        """
        Given a new set of documents (supplied as texts or doc_topics), the score method
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

        """
        # error-checks
        if texts is not None and doc_topics is not None:
            raise ValueError('texts is mutually-exclusive with doc_topics')
        if texts is None and doc_topics is None:
            raise ValueError('One of texts or doc_topics is required.')
        if texts is not None and type(texts) not in [list, np.ndarray]:
            raise ValueError('texts must be either a list or numpy ndarray')
        if  doc_topics is not None and type(doc_topics) not in [np.ndarray]:
            raise ValueError('doc_topics must be a np.ndarray')

        x_test = doc_topics
        if texts:
            x_test = self.predict(texts)
        return self.scorer.decision_function(x_test)


    def search(self, query, topic_ids=[], doc_ids=[], case_sensitive=False):
        """
        search documents for query string.
        Args:
            query(str):  the word or phrase to search
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
            case_sensitive(bool):  If True, case sensitive search
        """

        # setup pattern
        if not case_sensitive: query = query.lower()
        pattern = re.compile(r'\b%s\b' % query)

        # retrive docs
        docs = self.get_docs(topic_ids=topic_ids, doc_ids=doc_ids)

        # search
        mb = master_bar(range(1))
        results = []
        for i in mb:
            for doc in progress_bar(docs, parent=mb):
                text = doc[0]
                if not case_sensitive: text = text.lower()
                matches = pattern.findall(text)
                if matches: results.append(doc)
            if self.verbose: mb.write('done.')
        return results



    def _rank_documents(self, 
                       texts,
                       doc_topics=None):
        """
        Rank documents by topic score.
        If topic_index is supplied, rank documents based on relevance to supplied topic.
        Otherwise, rank all texts by their highest topic score (for any topic).
        Args:
            texts(list of str): list of document texts.
            doc_topics(ndarray): pre-computed topic distribution for each document
                                 If None, re-computed from texts.
                              
        Returns:
            dict of lists: each element in list is a tuple of (doc_index, topic_index, score)
            ... where doc_index is an index into either texts 
        """
        if doc_topics is not None:
            X_topics = doc_topics
        else:
            if self.verbose: print('transforming texts to topic space...')
            X_topics = self.predict(texts)
        topics = np.argmax(X_topics, axis=1)
        scores = np.amax(X_topics, axis=1)
        doc_ids = np.array([i for i, x in enumerate(texts)])
        result = list(zip(texts, doc_ids, topics, scores))
        if self.verbose: print('done.')
        result = sorted(result, key=lambda x: x[-1], reverse=True)
        result_dict = {}
        for r in result:
            text = r[0]
            doc_id = r[1]
            topic_id = r[2]
            score = r[3]
            lst = result_dict.get(topic_id, [])
            lst.append((text, doc_id, score))
            result_dict[topic_id] = lst
        return result_dict


    def _harden_topics(self, X_topics):
        """
        Transforms soft-clustering to hard-clustering
        """
        max_topics = []
        for i in range(X_topics.shape[0]):
            max_topics.append(X_topics[i].argmax())
        X_topics = np.array(max_topics)
        return X_topics


    def _check_build(self):
        self._check_model()
        if self.topic_dict is None: 
            raise Exception('Must call build() method.')

    def _check_scorer(self):
        if self.scorer is None:
            raise Exception('Must call train_scorer()')

    def _check_recommender(self):
        if self.recommender is None:
            raise Exception('Must call train_recommender()')


    def _check_model(self):
        if self.model is None or self.vectorizer is None:
            raise Exception('Must call train()')



    def save(self, fname):
        """
        save TopicModel object
        """

        
        with open(fname+'.tm_vect', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(fname+'.tm_model', 'wb') as f:
            pickle.dump(self.model, f)
        params = {'n_topics': self.n_topics,
                  'n_features': self.n_features,
                  'verbose': self.verbose}
        with open(fname+'.tm_params', 'wb') as f:
            pickle.dump(params, f)

        return

get_topic_model = TopicModel 























