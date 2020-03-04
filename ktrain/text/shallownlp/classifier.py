from .imports import *
from . import utils as U



__all__ = ['NBSVM']


class Classifier:
    def __init__(self, model=None):
        """
        instantiate a classifier with an optional previously-saved model
        """
        self.model = None


    def create_model(self, ctype, texts, hp_dict={}, ngram_range=(1,3), binary=True):
        """
        create a model
        Args:
          ctype(str): one of {'nbsvm', 'logreg', 'sgdclassifier'}
          texts(list): list of texts
          hp_dict(dict): dictionary of hyperparameters to use for the ctype selected.
                         hp_dict can also be used to supply arguments to CountVectorizer
          ngram_range(tuple): default ngram_range.
                              overridden if 'ngram_range' in hp_dict
          binary(bool): default value for binary argument to CountVectorizer.
                        overridden if 'binary' key in hp_dict

        """
        lang = U.detect_lang(texts)
        if U.is_chinese(lang):
            token_pattern = r'(?u)\b\w+\b'
        else:
            token_pattern = r'\w+|[%s]' % string.punctuation
        if ctype == 'nbsvm':
            clf = NBSVM(C=hp_dict.get('C', 0.01), 
                        alpha=hp_dict.get('alpha', 0.75), 
                        beta=hp_dict.get('beta', 0.25), 
                        fit_intercept=hp_dict.get('fit_intercept', False))
        elif ctype=='logreg':
            clf = LogisticRegression(C=hp_dict.get('C', 0.1), 
                                     dual=hp_dict.get('dual', True),
                                     penalty=hp_dict.get('penalty', 'l2'),
                                     tol=hp_dict.get('tol', 1e-4),
                                     intercept_scaling=hp_dict.get('intercept_scaling', 1),
                                     solver=hp_dict.get('solver', 'liblinear'),
                                     max_iter=hp_dict.get('max_iter', 100),
                                     multi_class=hp_dict.get('multi_class', 'auto'),
                                     warm_start=hp_dict.get('warm_start', False),
                                     n_jobs=hp_dict.get('n_jobs', None),
                                     l1_ratio=hp_dict.get('l1_ratio', None),
                                     random_state=hp_dict.get('random_state', 42),
                                     class_weight=hp_dict.get('class_weight', None)
                                     )
        elif ctype == 'sgdclassifier':
            clf = SGDClassifier(loss=hp_dict.get('loss', 'hinge'), 
                                penalty=hp_dict.get('penalty', 'l2'), 
                                alpha=hp_dict.get('alpha', 1e-3), 
                                random_state=hp_dict.get('random_state', 42), 
                                max_iter=hp_dict.get('max_iter', 5),  # scikit-learn default is 1000
                                tol=hp_dict.get('tol', None),
                                l1_ratio=hp_dict.get('l1_ratio', 0.15),
                                fit_intercept=hp_dict.get('fit_intercept', True),
                                episilon=hp_dict.get('epsilon', 0.1),
                                n_jobs=hp_dict.get('n_jobs', None),
                                learning_rate=hp_dict.get('learning_rate', 'optimal'),
                                eta0=hp_dict.get('eta0', 0.0),
                                power_t=hp_dict.get('power_t', 0.5),
                                early_stopping=hp_dict.get('early_stopping', False),
                                validation_fraction=hp_dict.get('validation_fraction', 0.1),
                                n_iter_no_change=hp_dict.get('n_iter_no_change', 5),
                                warm_start=hp_dict.get('warm_start', False),
                                average=hp_dict.get('average', False),
                                class_weight=hp_dict.get('class_weight', None))
        else:
            raise ValueError('Unknown ctype: %s' % (ctype))

        self.model = Pipeline([ ('vect', CountVectorizer(ngram_range=hp_dict.get('ngram_range', ngram_range), 
                                                         binary=hp_dict.get('binary', binary), 
                                                         token_pattern=token_pattern,
                                                         max_features=hp_dict.get('max_features', None),
                                                         max_df=hp_dict.get('max_df', 1.0),
                                                         min_df=hp_dict.get('min_df', 1),
                                                         stop_words=hp_dict.get('stop_words', None),
                                                         lowercase=hp_dict.get('lowercase', True),
                                                         strip_accents=hp_dict.get('strip_accents', None),
                                                         encoding=hp_dict.get('encoding', 'utf-8')
                                                         )),
                              ('clf', clf) ])
        return


    @classmethod
    def load_texts_from_folder(cls, folder_path, 
                              subfolders=None, 
                              shuffle=True,
                              encoding=None):
        """
        load text files from folder

        Args:
          folder_path(str): path to folder containing documents
                            The supplied folder should contain a subfolder
                            for each category, which will be used as the class label
          subfolders(list): list of subfolders under folder_path to consider
                            Example: If folder_path contains subfolders pos, neg, and 
                            unlabeled, then unlabeled folder can be ignored by
                            setting subfolders=['pos', 'neg']
          shuffle(bool):  If True, list of texts will be shuffled
          encoding(str): encoding to use.  default:None (auto-detected)
        Returns:
          tuple: (texts, labels, label_names)
        """
        bunch = load_files(folder_path, categories=subfolders, shuffle=shuffle)
        texts = bunch.data
        labels = bunch.target
        label_names = bunch.target_names
        #print('target names:')
        #for idx, label_name in enumerate(bunch.target_names):
            #print('\t%s:%s' % (idx, label_name))

        # decode based on supplied encoding
        if encoding is None:
            encoding = U.detect_encoding(texts)
            if encoding != 'utf-8':
                print('detected encoding: %s' % (encoding))

        try:
            texts = [text.decode(encoding) for text in texts]
        except:
            print('Decoding with %s failed 1st attempt - using %s with skips' % (encoding,
                                                                                 encoding))
            texts = U.decode_by_line(texts, encoding=encoding)
        return (texts, labels, label_names)



    @classmethod
    def load_texts_from_csv(cls, csv_filepath, text_column='text', label_column='label',
                            sep=',', encoding=None):
        """
        load text files from csv file
        CSV should have at least two columns.
        Example:
        Text               | Label
        I love this movie. | positive
        I hated this movie.| negative


        Args:
          csv_filepath(str): path to CSV file
          text_column(str): name of column containing the texts. default:'text'
          label_column(str): name of column containing the labels in string format
                             default:'label'
          sep(str): character that separates columns in CSV. default:','
          encoding(str): encoding to use. default:None (auto-detected)
        Returns:
          tuple: (texts, labels, label_names)
        """
        if encoding is None:
            with open(csv_filepath, 'rb') as f:
                encoding = U.detect_encoding([f.read()])
                if encoding != 'utf-8':
                    print('detected encoding: %s (if wrong, set manually)' % (encoding))
        import pandas as pd
        df = pd.read_csv(csv_filepath, encoding=encoding, sep=sep)
        texts = df[text_column].fillna('fillna').values
        labels = df[label_column].values
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        return (texts, labels, le.classes_)


    def fit(self, x_train, y_train, ctype='logreg'):
        """
        train a classifier
        Args:
          x_train(list or np.ndarray):  training texts
          y_train(np.ndarray):  training labels
          ctype(str):  One of {'logreg', 'nbsvm', 'sgdclassifier'}.  default:nbsvm
        """
        lang = U.detect_lang(x_train)
        if U.is_chinese(lang):
            x_train = U.split_chinese(x_train)
        if self.model is None:
            self.create_model(ctype, x_train)
        self.model.fit(x_train, y_train)
        return self



    def predict(self, x_test, return_proba=False):
        """
        make predictions on text data
        Args:
          x_test(list or np.ndarray or str): array of texts on which to make predictions or a string representing text
        """
        if return_proba and not hasattr(self.model['clf'], 'predict_proba'): 
            raise ValueError('%s does not support predict_proba' % (type(self.model['clf']).__name__))
        if isinstance(x_test, str): x_test = [x_test]
        lang = U.detect_lang(x_test)
        if U.is_chinese(lang): x_test = U.split_chinese(x_test)
        if self.model is None: raise ValueError('model is None - call fit or load to set the model')
        if return_proba:
            predicted = self.model.predict_proba(x_test)
        else:
            predicted = self.model.predict(x_test)
        if len(predicted) == 1: predicted = predicted[0]
        return predicted


    def predict_proba(self, x_test):
        """
        predict_proba
        """
        return self.predict(x_test, return_proba=True)


    def evaluate(self, x_test, y_test):
        """
        evaluate
        Args:
          x_test(list or np.ndarray):  training texts
          y_test(np.ndarray):  training labels
        """
        predicted = self.predict(x_test)
        return np.mean(predicted == y_test)


    def save(self, filename):
        """
        save model
        """
        dump(self.model, filename)


    def load(self, filename):
        """
        load model
        """
        self.model = load(filename)

    def grid_search(self, params, x_train, y_train, n_jobs=-1):
        """
        Performs grid search to find optimal set of hyperparameters
        Args:
          params (dict):  A dictionary defining the space of the search.
                          Example for finding optimal value of alpha in NBSVM:
                        parameters = {
                                      #'clf__C': (1e0, 1e-1, 1e-2),
                                      'clf__alpha': (0.1, 0.2, 0.4, 0.5, 0.75, 0.9, 1.0),
                                      #'clf__fit_intercept': (True, False),
                                      #'clf__beta' : (0.1, 0.25, 0.5, 0.9) 
                                      }
          n_jobs(int): number of jobs to run in parallel.  default:-1 (use all processors)
        """
        gs_clf = GridSearchCV(self.model, params, n_jobs=n_jobs)
        gs_clf = gs_clf.fit(x_train, y_train)
	#gs_clf.best_score_                                  
        for param_name in sorted(params.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        return





class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):

    def __init__(self, alpha=1, C=1, beta=0.25, fit_intercept=False):
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            coef_, intercept_ = self._fit_binary(X, y)
            self.coef_ = coef_
            self.intercept_ = intercept_
        else:
            coef_, intercept_ = zip(*[
                self._fit_binary(X, y == class_)
                for class_ in self.classes_
            ])
            self.coef_ = np.concatenate(coef_)
            self.intercept_ = np.array(intercept_).flatten()
        return self

    def _fit_binary(self, X, y):
        p = np.asarray(self.alpha + X[y == 1].sum(axis=0)).flatten()
        q = np.asarray(self.alpha + X[y == 0].sum(axis=0)).flatten()
        r = np.log(p/np.abs(p).sum()) - np.log(q/np.abs(q).sum())
        b = np.log((y == 1).sum()) - np.log((y == 0).sum())

        if isinstance(X, spmatrix):
            indices = np.arange(len(r))
            r_sparse = coo_matrix(
                (r, (indices, indices)),
                shape=(len(r), len(r))
            )
            X_scaled = X * r_sparse
        else:
            X_scaled = X * r

        lsvc = LinearSVC(
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=10000
        ).fit(X_scaled, y)

        mean_mag =  np.abs(lsvc.coef_).mean()

        coef_ = (1 - self.beta) * mean_mag * r + \
                self.beta * (r * lsvc.coef_)

        intercept_ = (1 - self.beta) * mean_mag * b + \
                     self.beta * lsvc.intercept_

        return coef_, intercept_

