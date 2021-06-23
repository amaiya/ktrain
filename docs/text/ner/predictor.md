Module ktrain.text.ner.predictor
================================

Classes
-------

`NERPredictor(model, preproc, batch_size=32)`
:   predicts  classes for string-representation of sentence

    ### Ancestors (in MRO)

    * ktrain.predictor.Predictor
    * abc.ABC

    ### Methods

    `get_classes(self)`
    :

    `merge_tokens(self, annotated_sentence, lang)`
    :

    `predict(self, sentences, return_proba=False, merge_tokens=False, custom_tokenizer=None)`
    :   Makes predictions for a string-representation of a sentence
        Args:
          sentences(list|str): either a single sentence as a string or a list of sentences
          return_proba(bool): If return_proba is True, returns probability distribution for each token
          merge_tokens(bool):  If True, tokens will be merged together by the entity
                               to which they are associated:
                               ('Paul', 'B-PER'), ('Newman', 'I-PER') becomes ('Paul Newman', 'PER')
          custom_tokenizer(Callable): If specified, sentence will be tokenized based on custom tokenizer
        
        Returns:
          list: If sentences is a string representation of single sentence:
                     list containing a tuple for each token in sentence
                IF sentences is a list of sentences:
                     list  of lists:  Each inner list represents a sentence and contains a tuple for each token in sentence