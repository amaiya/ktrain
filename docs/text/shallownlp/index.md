Module ktrain.text.shallownlp
=============================

Sub-modules
-----------
* ktrain.text.shallownlp.classifier
* ktrain.text.shallownlp.imports
* ktrain.text.shallownlp.ner
* ktrain.text.shallownlp.searcher
* ktrain.text.shallownlp.utils
* ktrain.text.shallownlp.version

Functions
---------

    
`extract_filenames(corpus_path, follow_links=False)`
:   

    
`find_arabic(s)`
:   

    
`find_chinese(s)`
:   

    
`find_russian(s)`
:   

    
`read_text(filename)`
:   

    
`search(query, doc, case_sensitive=False, keys=[], progress=False)`
:   

    
`sent_tokenize(text)`
:   segment text into sentences

Classes
-------

`Classifier(model=None)`
:   instantiate a classifier with an optional previously-saved model

    ### Static methods

    `load_texts_from_csv(csv_filepath, text_column='text', label_column='label', sep=',', encoding=None)`
    :   load text files from csv file
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

    `load_texts_from_folder(folder_path, subfolders=None, shuffle=True, encoding=None)`
    :   load text files from folder
        
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

    ### Methods

    `create_model(self, ctype, texts, hp_dict={}, ngram_range=(1, 3), binary=True)`
    :   create a model
        Args:
          ctype(str): one of {'nbsvm', 'logreg', 'sgdclassifier'}
          texts(list): list of texts
          hp_dict(dict): dictionary of hyperparameters to use for the ctype selected.
                         hp_dict can also be used to supply arguments to CountVectorizer
          ngram_range(tuple): default ngram_range.
                              overridden if 'ngram_range' in hp_dict
          binary(bool): default value for binary argument to CountVectorizer.
                        overridden if 'binary' key in hp_dict

    `evaluate(self, x_test, y_test)`
    :   evaluate
        Args:
          x_test(list or np.ndarray):  training texts
          y_test(np.ndarray):  training labels

    `fit(self, x_train, y_train, ctype='logreg')`
    :   train a classifier
        Args:
          x_train(list or np.ndarray):  training texts
          y_train(np.ndarray):  training labels
          ctype(str):  One of {'logreg', 'nbsvm', 'sgdclassifier'}.  default:nbsvm

    `grid_search(self, params, x_train, y_train, n_jobs=-1)`
    :   Performs grid search to find optimal set of hyperparameters
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

    `load(self, filename)`
    :   load model

    `predict(self, x_test, return_proba=False)`
    :   make predictions on text data
        Args:
          x_test(list or np.ndarray or str): array of texts on which to make predictions or a string representing text

    `predict_proba(self, x_test)`
    :   predict_proba

    `save(self, filename)`
    :   save model

`NER(lang='en', predictor_path=None)`
:   pretrained NER.
    Only English and Chinese are currenty supported.
    
    Args:
      lang(str): Currently, one of {'en', 'zh', 'ru'}: en=English , zh=Chinese, or ru=Russian

    ### Methods

    `predict(self, texts, merge_tokens=True, batch_size=32)`
    :   Extract named entities from supplied text
        
        Args:
          texts (list of str or str): list of texts to annotate
          merge_tokens(bool):  If True, tokens will be merged together by the entity
                               to which they are associated:
                               ('Paul', 'B-PER'), ('Newman', 'I-PER') becomes ('Paul Newman', 'PER')
          batch_size(int):    Batch size to use for predictions (default:32)

`Searcher(queries, lang=None)`
:   Search for keywords in text documents
    
    Args:
      queries(list of str): list of chinese text queries
      lang(str): language of queries.  default:None --> auto-detected

    ### Methods

    `search(self, docs, case_sensitive=False, keys=[], min_matches=1, progress=True)`
    :   executes self.queries on supplied list of documents
        Args:
          docs(list of str): list of chinese texts
          case_sensitive(bool):  If True, case sensitive search
          keys(list): list keys for supplied docs (e.g., file paths).
                      default: key is index in range(len(docs))
          min_matches(int): results must have at least these many word matches
          progress(bool): whether or not to show progress bar
        Returns:
          list of tuples of results of the form:
            (key, query, no. of matches)
          For Chinese, no. of matches will be number of unique Jieba-extracted character sequences that match