Module ktrain.text.zsl.core
===========================

Functions
---------

    
`list2chunks(a, n)`
:   

Classes
-------

`ZeroShotClassifier(model_name='facebook/bart-large-mnli', device=None)`
:   interface to Zero Shot Topic Classifier
    
    ZeroShotClassifier constructor
    
    Args:
      model_name(str): name of a BART NLI model
      device(str): device to use (e.g., 'cuda', 'cpu')

    ### Methods

    `predict(self, docs, labels=[], include_labels=False, multilabel=True, max_length=512, batch_size=8, nli_template='This text is about {}.', topic_strings=[])`
    :   This method performs zero-shot text classification using Natural Language Inference (NLI).
        Args:
          docs(list|str): text of document or list of texts
          labels(list): a list of strings representing topics of your choice
                        Example:
                          labels=['political science', 'sports', 'science']
          include_labels(bool): If True, will return topic labels along with topic probabilities
          multilabel(bool): If True, labels are considered independent and multiple labels can predicted true for document and be close to 1.
                            If False, scores are normalized such that probabilities sum to 1.
          max_length(int): truncate long documents to this many tokens
          batch_size(int): batch_size to use. default:8
                           Increase this value to speed up predictions - especially
                           if len(topic_strings) is large.
          nli_template(str): labels are inserted into this template for use as hypotheses in natural language inference
          topic_strings(list): alias for labels parameter for backwards compatibility
        Returns:
          inferred probabilities or list of inferred probabilities if doc is list