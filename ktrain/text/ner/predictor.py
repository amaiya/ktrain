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


    def predict(self, sentences, return_proba=False, merge_tokens=False, custom_tokenizer=None):
        """
        Makes predictions for a string-representation of a sentence
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
        """
        is_array = not isinstance(sentences, str)
        if not isinstance(sentences, (str, list)):
            raise ValueError('Param sentence must be either string-representation of a sentence or a list of sentence strings.')
        if return_proba and merge_tokens:
            raise ValueError('return_proba and merge_tokens are mutually exclusive with one another.')
        if isinstance(sentences, str): sentences = [sentences]
        lang = TU.detect_lang(sentences)

        # batchify
        num_chunks = math.ceil(len(sentences)/self.batch_size)
        batches = U.list2chunks(sentences, n=num_chunks)

        # process batches
        results = []
        for batch in batches:
            nerseq = self.preproc.preprocess(batch, lang=lang, custom_tokenizer=custom_tokenizer)
            if not nerseq.prepare_called:
                nerseq.prepare()
            nerseq.batch_size = len(batch)
            x_true, _ = nerseq[0]
            lengths = nerseq.get_lengths(0)
            y_pred = self.model.predict_on_batch(x_true)
            y_labels = self.preproc.p.inverse_transform(y_pred, lengths)
            if return_proba:
                try:
                    probs = np.max(y_pred, axis=2)
                except:
                    probs = y_pred[0].numpy().tolist() # TODO: remove after confirmation (#316)
                for x, y, prob in zip(nerseq.x, y_labels, probs):
                    result = [(x[i], y[i], prob[i]) for i in range(len(x))]
                    results.append(result)
            else:
                for x,y in zip(nerseq.x, y_labels):
                    result =  list(zip(x,y))
                    if merge_tokens:
                        result = self.merge_tokens(result, lang)
                    results.append(result)
        if not is_array: results = results[0]
        return results



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
        if current_token and current_tag:
            entities.append((current_token, current_tag))
        return entities

