from ... import utils as U
from ...imports import *
from ...predictor import Predictor
from .. import textutils as TU
from .preprocessor import NERPreprocessor


class NERPredictor(Predictor):
    """
    predicts  classes for string-representation of sentence
    """

    def __init__(self, model, preproc, batch_size=U.DEFAULT_BS):
        if not isinstance(model, keras.Model):
            raise ValueError("model must be of instance keras.Model")
        if not isinstance(preproc, NERPreprocessor):
            # if type(preproc).__name__ != 'NERPreprocessor':
            raise ValueError("preproc must be a NERPreprocessor object")
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()
        self.batch_size = batch_size

    def get_classes(self):
        return self.c

    def predict(
        self,
        sentences,
        return_proba=False,
        merge_tokens=False,
        custom_tokenizer=None,
        return_offsets=False,
    ):
        """
        ```
        Makes predictions for a string-representation of a sentence
        Args:
          sentences(list|str): either a single sentence as a string or a list of sentences
          return_proba(bool): If return_proba is True, returns probability distribution for each token
          merge_tokens(bool):  If True, tokens will be merged together by the entity
                               to which they are associated:
                               ('Paul', 'B-PER'), ('Newman', 'I-PER') becomes ('Paul Newman', 'PER')
          custom_tokenizer(Callable): If specified, sentence will be tokenized based on custom tokenizer
          return_offsets(bool): If True, will return the chracter offsets in the results [experimental]

        Returns:
          list: If sentences is a string representation of single sentence:
                     list containing a tuple for each token in sentence
                IF sentences is a list of sentences:
                     list  of lists:  Each inner list represents a sentence and contains a tuple for each token in sentence
                If return_proba and return_offsets are both True, then tuples are of the form:  (token, label, probability, character offsets)
        ```
        """
        is_array = not isinstance(sentences, str)
        if not isinstance(sentences, (str, list)):
            raise ValueError(
                "Param sentence must be either string-representation of a sentence or a list of sentence strings."
            )
        # if return_proba and merge_tokens:
        #     raise ValueError(
        #         "return_proba and merge_tokens are mutually exclusive with one another."
        #     )
        if isinstance(sentences, str):
            sentences = [sentences]
        lang = TU.detect_lang(sentences)

        # batchify
        num_chunks = math.ceil(len(sentences) / self.batch_size)
        batches = U.list2chunks(sentences, n=num_chunks)

        # process batches
        results = []
        for batch in batches:
            nerseq = self.preproc.preprocess(
                batch, lang=lang, custom_tokenizer=custom_tokenizer
            )
            if not nerseq.prepare_called:
                nerseq.prepare()
            nerseq.batch_size = len(batch)
            x_true, _ = nerseq[0]
            lengths = nerseq.get_lengths(0)
            y_pred = self.model.predict_on_batch(x_true)
            y_labels = self.preproc.p.inverse_transform(y_pred, lengths)
            # TODO: clean this up
            if return_proba:
                try:
                    probs = np.max(y_pred, axis=2)
                except:
                    probs = (
                        y_pred[0].numpy().tolist()
                    )  # TODO: remove after confirmation (#316)
                for i, (x, y, prob) in enumerate(zip(nerseq.x, y_labels, probs)):
                    if return_offsets:
                        offsets = TU.extract_offsets(
                            sentences[i], tokens=[entry[0] for entry in x]
                        )
                        result = [
                            (
                                x[i],
                                y[i],
                                prob[i],
                                (offsets[i]["start"], offsets[i]["end"]),
                            )
                            for i in range(len(x))
                        ]
                    else:
                        result = [(x[i], y[i], prob[i]) for i in range(len(x))]
                    if merge_tokens:
                        result = self.merge_tokens(result, lang, True)
                    results.append(result)
            else:
                for i, (x, y) in enumerate(zip(nerseq.x, y_labels)):
                    if return_offsets:
                        offsets = TU.extract_offsets(
                            sentences[i], tokens=[entry[0] for entry in x]
                        )
                        result = list(
                            zip(x, y, [(o["start"], o["end"]) for o in offsets])
                        )
                    else:
                        result = list(zip(x, y))
                    if merge_tokens:
                        result = self.merge_tokens(result, lang, False)
                    results.append(result)
        if not is_array:
            results = results[0]
        return results

    def merge_tokens(self, annotated_sentence, lang, return_proba):
        if TU.is_chinese(
            lang, strict=False
        ):  # strict=False: workaround for langdetect bug on short chinese texts
            sep = ""
        else:
            sep = " "

        current_token = ""
        current_tag = ""
        prob_list = []
        entities = []
        start = None
        last_end = None

        for tup in annotated_sentence:
            token = tup[0]
            entity = tup[1]
            if return_proba:
                prob = tup[2]
                offsets = tup[3] if len(tup) > 3 else None
            else:
                offsets = tup[2] if len(tup) > 2 else None
            tag = entity.split("-")[1] if "-" in entity else None
            prefix = entity.split("-")[0] if "-" in entity else None
            # not within entity
            if tag is None and not current_token:
                continue
            # beginning of entity
            # elif tag and prefix=='B':
            elif tag and (prefix == "B" or prefix == "I" and not current_token):
                if current_token:  # consecutive entities
                    entities.append(
                        self._build_merge_tuple(
                            current_token, current_tag, start, last_end, prob_list
                        )
                    )
                    prob_list = []
                    current_token = ""
                    current_tag = None
                    start, end = None, None
                current_token = token
                current_tag = tag
                start = offsets[0] if offsets else None
                last_end = offsets[1] if offsets else None
                if return_proba:
                    prob_list.append(prob)
            # end of entity
            elif tag is None and current_token:
                entities.append(
                    self._build_merge_tuple(
                        current_token, current_tag, start, last_end, prob_list
                    )
                )
                prob_list = []
                current_token = ""
                current_tag = None
                continue
            # within entity
            elif tag and current_token:  # prefix I
                current_token = current_token + sep + token
                current_tag = tag
                last_end = offsets[1] if offsets else None
                if return_proba:
                    prob_list.append(prob)
        if current_token and current_tag:
            entities.append(
                self._build_merge_tuple(
                    current_token, current_tag, start, last_end, prob_list
                )
            )
        return entities

    def _build_merge_tuple(
        self, current_token, current_tag, start=None, end=None, prob_list=[]
    ):
        entry = [current_token, current_tag]
        if start is not None and end is not None:
            entry.append((start, end))
        if prob_list:
            entry.append(np.mean(prob_list))
        return tuple(entry)

    def _save_preproc(self, fpath):
        # ensure transformers embedding model is saved in a subdirectory
        p = self.preproc.p
        hf_dir = os.path.join(fpath, "hf")
        if p.te is not None:
            os.makedirs(hf_dir, exist_ok=True)
            p.te.model.save_pretrained(hf_dir)
            p.te.tokenizer.save_pretrained(hf_dir)
            p.te.config.save_pretrained(hf_dir)
            p.te_model = hf_dir

        # save preproc
        with open(os.path.join(fpath, U.PREPROC_NAME), "wb") as f:
            pickle.dump(self.preproc, f)
        return
