from ...imports import *
from ...predictor import Predictor
from .preprocessor import NERPreprocessor
from ... import utils as U
from .. import textutils as TU

class NERPredictor(Predictor):
    """
    predicts  classes for string-representation of sentence
    """

    def __init__(self, model, preproc, batch_size=U.DEFAULT_BS):

        if not isinstance(model, Model):
            raise ValueError('model must be of instance Model')
        if not isinstance(preproc, NERPreprocessor):
        #if type(preproc).__name__ != 'NERPreprocessor':
            raise ValueError('preproc must be a NERPreprocessor object')
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()
        self.batch_size = batch_size 


    def get_classes(self):
        return self.c


    def predict(self, sentence, return_proba=False, merge_tokens=False, custom_tokenizer=None):
        """
        Makes predictions for a string-representation of a sentence
        Args:
          sentence(str): sentence of text
          return_proba(bool): If return_proba is True, returns probability distribution for each token
          merge_tokens(bool):  If True, tokens will be merged together by the entity
                               to which they are associated:
                               ('Paul', 'B-PER'), ('Newman', 'I-PER') becomes ('Paul Newman', 'PER')
          custom_tokenizer(Callable): If specified, sentence will be tokenized based on custom tokenizer

        Returns:
          list: list of tuples representing each token.
        """
        if not isinstance(sentence, str):
            raise ValueError('Param sentence must be a string-representation of a sentence')
        if return_proba and merge_tokens:
            raise ValueError('return_proba and merge_tokens are mutually exclusive with one another.')
        lang = TU.detect_lang([sentence])
        nerseq = self.preproc.preprocess([sentence], lang=lang, custom_tokenizer=custom_tokenizer)
        if not nerseq.prepare_called:
            nerseq.prepare()
        nerseq.batch_size = self.batch_size
        x_true, _ = nerseq[0]
        lengths = nerseq.get_lengths(0)
        y_pred = self.model.predict_on_batch(x_true)
        y_labels = self.preproc.p.inverse_transform(y_pred, lengths)
        y_labels = y_labels[0]
        if return_proba:
            #probs = np.max(y_pred, axis=2)[0]
            y_pred = y_pred[0].numpy().tolist()
            return list(zip(nerseq.x[0], y_labels, y_pred))
        else:
            result =  list(zip(nerseq.x[0], y_labels))
            if merge_tokens:
                result = self.merge_tokens(result, lang)
            return result


    def merge_tokens(self, annotated_sentence, lang):

        if TU.is_chinese(lang, strict=False): # strict=False: workaround for langdetect bug on short chinese texts
            sep = ''
        else:
            sep = ' '

        current_token = ""
        current_tag = ""
        entities = []

        for tup in annotated_sentence:
            token = tup[0]
            entity = tup[1]
            tag = entity.split('-')[1] if '-' in entity else None
            prefix = entity.split('-')[0] if '-' in entity else None
            # not within entity
            if tag is None and not current_token:
                continue
            # beginning of entity
            #elif tag and prefix=='B':
            elif tag and (prefix=='B' or prefix=='I' and not current_token):
                if current_token: # consecutive entities
                    entities.append((current_token, current_tag))
                    current_token = ""
                    current_tag = None
                current_token = token
                current_tag = tag
            # end of entity
            elif tag is None and current_token:
                entities.append((current_token, current_tag))
                current_token = ""
                current_tag = None
                continue
            # within entity
            elif tag and current_token:  #  prefix I
                current_token = current_token + sep + token
                current_tag = tag
        return entities

