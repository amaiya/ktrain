Module ktrain.text.ner.data
===========================

Functions
---------

    
`entities_from_array(x_train, y_train, x_test=None, y_test=None, use_char=False, val_pct=0.1, verbose=1)`
:   Load entities from arrays
    Args:
      x_train(list): list of list of entity tokens for training
                     Example: x_train = [['Hello', 'world'], ['Hello', 'Cher'], ['I', 'love', 'Chicago']]
      y_train(list): list of list of tokens representing entity labels
                     Example:  y_train = [['O', 'O'], ['O', 'B-PER'], ['O', 'O', 'B-LOC']]
      x_test(list): list of list of entity tokens for validation 
                     Example: x_train = [['Hello', 'world'], ['Hello', 'Cher'], ['I', 'love', 'Chicago']]
      y_test(list): list of list of tokens representing entity labels
                     Example:  y_train = [['O', 'O'], ['O', 'B-PER'], ['O', 'O', 'B-LOC']]
     use_char(bool):    If True, data will be preprocessed to use character embeddings  in addition to word embeddings
     val_pct(float):  percentage of training to use for validation if no validation data is supplied
     verbose (boolean): verbosity

    
`entities_from_conll2003(train_filepath, val_filepath=None, use_char=False, encoding=None, val_pct=0.1, verbose=1)`
:   Loads sequence-labeled data from a file in CoNLL2003 format.

    
`entities_from_df(train_df, val_df=None, word_column='Word', tag_column='Tag', sentence_column='SentenceID', use_char=False, val_pct=0.1, verbose=1)`
:   Load entities from pandas DataFrame
    Args:
      train_df(pd.DataFrame): training data
      val_df(pdf.DataFrame): validation data
      word_column(str): name of column containing the text
      tag_column(str): name of column containing lael
      sentence_column(str): name of column containing Sentence IDs
      use_char(bool):    If True, data will be preprocessed to use character embeddings  in addition to word embeddings
      verbose (boolean): verbosity

    
`entities_from_gmb(train_filepath, val_filepath=None, use_char=False, word_column='Word', tag_column='Tag', sentence_column='SentenceID', encoding=None, val_pct=0.1, verbose=1)`
:   Loads sequence-labeled data from text file in the  Groningen
    Meaning Bank  (GMB) format.

    
`entities_from_txt(train_filepath, val_filepath=None, use_char=False, word_column='Word', tag_column='Tag', sentence_column='SentenceID', data_format='conll2003', encoding=None, val_pct=0.1, verbose=1)`
:   Loads sequence-labeled data from comma or tab-delmited text file.
    Format of file is either the CoNLL2003 format or Groningen Meaning
    Bank (GMB) format - specified with data_format parameter.
    
    In both formats, each word appars on a separate line along with
    its associated tag (or label).  
    The last item on each line should be the tag or label assigned to word.
    
    In the CoNLL2003 format, there is an empty line after
    each sentence.  In the GMB format, sentences are deliniated
    with a third column denoting the Sentence ID.
    
    
    
    More information on CoNLL2003 format: 
       https://www.aclweb.org/anthology/W03-0419
    
    CoNLL Example (each column is typically separated by space or tab)
    and  no column headings:
    
       Paul     B-PER
       Newman   I-PER
       is       O
       a        O
       great    O
       actor    O
       !        O
    
    More information on GMB format:
    Refer to ner_dataset.csv on Kaggle here:
       https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/version/2
    
    GMB example (each column separated by comma or tab)
    with column headings:
    
      SentenceID   Word     Tag    
      1            Paul     B-PER
      1            Newman   I-PER
      1            is       O
      1            a        O
      1            great    O
      1            actor    O
      1            !        O
    
    
    Args:
        train_filepath(str): file path to training CSV
        val_filepath (str): file path to validation dataset
        use_char(bool):    If True, data will be preprocessed to use character embeddings in addition to word embeddings
        word_column(str): name of column containing the text
        tag_column(str): name of column containing lael
        sentence_column(str): name of column containing Sentence IDs
        data_format(str): one of colnll2003 or gmb
                          word_column, tag_column, and sentence_column
                          ignored if 'conll2003'
        encoding(str): the encoding to use.  If None, encoding is discovered automatically
        val_pct(float): Proportion of training to use for validation.
        verbose (boolean): verbosity