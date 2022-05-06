# -*- coding: utf-8 -*-
"""
Preprocessors.
"""

from .... import utils as U
from ....imports import *
from .utils import Vocabulary

try:
    from allennlp.modules.elmo import Elmo, batch_to_ids

    ALLENNLP_INSTALLED = True
except:
    ALLENNLP_INSTALLED = False


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


def normalize_number(text):
    return re.sub(r"[0-9０１２３４５６７８９]", r"0", text)


class IndexTransformer(BaseEstimator, TransformerMixin):
    """Convert a collection of raw documents to a document id matrix.

    Attributes:
        _use_char: boolean. Whether to use char feature.
        _num_norm: boolean. Whether to normalize text.
        _word_vocab: dict. A mapping of words to feature indices.
        _char_vocab: dict. A mapping of chars to feature indices.
        _label_vocab: dict. A mapping of labels to feature indices.
    """

    def __init__(
        self,
        lower=True,
        num_norm=True,
        use_char=True,
        initial_vocab=None,
        use_elmo=False,
    ):
        """Create a preprocessor object.

        Args:
            lower: boolean. Whether to convert the texts to lowercase.
            use_char: boolean. Whether to use char feature.
            num_norm: boolean. Whether to normalize text.
            initial_vocab: Iterable. Initial vocabulary for expanding word_vocab.
            use_elmo: If True, will generate contextual English Elmo embeddings
        """
        self._num_norm = num_norm
        self._use_char = use_char
        self._word_vocab = Vocabulary(lower=lower)
        self._char_vocab = Vocabulary(lower=False)
        self._label_vocab = Vocabulary(lower=False, unk_token=False)

        if initial_vocab:
            self._word_vocab.add_documents([initial_vocab])
            self._char_vocab.add_documents(initial_vocab)

        self.elmo = None  # elmo embedding model
        self.use_elmo = False
        self.te = None  # transformer embedding model
        self.te_layers = U.DEFAULT_TRANSFORMER_LAYERS
        self.te_model = None
        self._blacklist = ["te", "elmo"]

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._blacklist}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "te_model"):
            self.te_model = None
        if not hasattr(self, "use_elmo"):
            self.use_elmo = False
        if not hasattr(self, "te_layers"):
            self.te_layers = U.DEFAULT_TRANSFORMER_LAYERS

        try:
            if self.te_model is not None:
                self.activate_transformer(self.te_model, layers=self.te_layers)
            else:
                self.te = None
        except:
            self.te = None  # set in predictor for support for air-gapped networks
        if self.use_elmo:
            self.activate_elmo()
        else:
            self.elmo = None

    def activate_elmo(self):
        if not ALLENNLP_INSTALLED:
            raise Exception(ALLENNLP_ERRMSG)

        if not hasattr(self, "elmo"):
            self.elmo = None
        if self.elmo is None:
            self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        self.use_elmo = True

    def activate_transformer(
        self, model_name, layers=U.DEFAULT_TRANSFORMER_LAYERS, force=False
    ):
        from ...preprocessor import TransformerEmbedding

        if not hasattr(self, "te"):
            self.te = None
        if self.te is None or self.te_model != model_name or force:
            self.te_model = model_name
            self.te = TransformerEmbedding(model_name, layers=layers)
        self.te_layers = layers

    def get_transformer_dim(self):
        if not self.transformer_is_activated():
            return None
        else:
            return self.te.embsize

    def elmo_is_activated(self):
        return self.elmo is not None

    def transformer_is_activated(self):
        return self.te is not None

    def fix_tokenization(
        self,
        X,
        Y,
        maxlen=U.DEFAULT_TRANSFORMER_MAXLEN,
        num_special=U.DEFAULT_TRANSFORMER_NUM_SPECIAL,
    ):
        """
        Should be called prior training
        """
        if not self.transformer_is_activated():
            return X, Y
        ids2tok = self.te.tokenizer.convert_ids_to_tokens
        encode = self.te.tokenizer.encode_plus
        new_X = []
        new_Y = []
        for i, x in enumerate(X):
            new_x = []
            new_y = []
            seq_len = 0
            for j, s in enumerate(x):
                # subtokens = ids2tok(encode(s, add_special_tokens=False))
                encoded_input = encode(
                    s, add_special_tokens=False, return_offsets_mapping=True
                )
                offsets = encoded_input["offset_mapping"]
                subtokens = encoded_input.tokens()
                token_len = len(subtokens)
                if (seq_len + token_len) > (maxlen - num_special):
                    break
                seq_len += token_len

                if len(s.split()) == 1:
                    hf_s = [s]
                else:
                    word_ids = encoded_input.word_ids()
                    hf_s = []
                    for k, subtoken in enumerate(subtokens):
                        word_id = word_ids[k]
                        currlen = len(hf_s)
                        if currlen == word_id + 1:
                            hf_s[word_id].append(offsets[k])
                        elif word_id + 1 > currlen:
                            hf_s.append([offsets[k]])
                    hf_s = [s[entry[0][0] : entry[-1][1]] for entry in hf_s]

                new_x.extend(hf_s)
                if Y is not None:
                    tag = Y[i][j]
                    new_y.extend([tag])
                    if len(hf_s) > 1:
                        new_tag = tag
                        if tag.startswith("B-"):
                            new_tag = "I-" + tag[2:]
                        new_y.extend([new_tag] * (len(hf_s) - 1))
            new_X.append(new_x)
            new_Y.append(new_y)
        new_Y = None if Y is None else new_Y
        return new_X, new_Y

    def fit(self, X, y):
        """Learn vocabulary from training set.

        Args:
            X : iterable. An iterable which yields either str, unicode or file objects.

        Returns:
            self : IndexTransformer.
        """
        self._word_vocab.add_documents(X)
        self._label_vocab.add_documents(y)
        if self._use_char:
            for doc in X:
                self._char_vocab.add_documents(doc)

        self._word_vocab.build()
        self._char_vocab.build()
        self._label_vocab.build()

        return self

    def transform(self, X, y=None):
        """Transform documents to document ids.

        Uses the vocabulary learned by fit.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.
            y : iterabl, label strings.

        Returns:
            features: document id matrix.
            y: label id matrix.
        """
        # re-instantiate TransformerEmbedding/Elmo if necessary since it is excluded from pickling
        if self.te_model is not None:
            self.activate_transformer(self.te_model, layers=self.te_layers)
        if self.use_elmo:
            self.activate_elmo()

        features = []

        word_ids = [self._word_vocab.doc2id(doc) for doc in X]
        word_ids = keras.preprocessing.sequence.pad_sequences(word_ids, padding="post")
        features.append(word_ids)

        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X]
            char_ids = pad_nested_sequences(char_ids)
            features.append(char_ids)

        if self.elmo is not None:
            if not ALLENNLP_INSTALLED:
                raise Exception(ALLENNLP_ERRMSG)

            character_ids = batch_to_ids(X)
            elmo_embeddings = self.elmo(character_ids)["elmo_representations"][1]
            elmo_embeddings = elmo_embeddings.detach().numpy()
            features.append(elmo_embeddings)

        if self.te is not None:
            transformer_embeddings = self.te.embed(X, word_level=True)
            features.append(transformer_embeddings)
            # print(f' | {X} | [trans_shape={transformer_embeddings.shape[1]} | word_id_shape={len(word_ids)}')

        if y is not None:
            y = [self._label_vocab.doc2id(doc) for doc in y]
            y = keras.preprocessing.sequence.pad_sequences(y, padding="post")
            y = keras.utils.to_categorical(y, self.label_size).astype(int)
            # In 2018/06/01, to_categorical is a bit strange.
            # >>> to_categorical([[1,3]], num_classes=4).shape
            # (1, 2, 4)
            # >>> to_categorical([[1]], num_classes=4).shape
            # (1, 4)
            # So, I expand dimensions when len(y.shape) == 2.
            y = y if len(y.shape) == 3 else np.expand_dims(y, axis=0)
            return features, y
        else:
            return features

    def fit_transform(self, X, y=None, **params):
        """Learn vocabulary and return document id matrix.

        This is equivalent to fit followed by transform.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.

        Returns:
            list : document id matrix.
            list: label id matrix.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y, lengths=None):
        """Return label strings.

        Args:
            y: label id matrix.
            lengths: sentences length.

        Returns:
            list: list of list of strings.
        """
        y = np.argmax(y, -1)
        inverse_y = [self._label_vocab.id2doc(ids) for ids in y]
        if lengths is not None:
            inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]

        return inverse_y

    @property
    def word_vocab_size(self):
        return len(self._word_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    @property
    def label_size(self):
        return len(self._label_vocab)

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p


def pad_nested_sequences(sequences, dtype="int32"):
    """Pads nested sequences to the same length.

    This function transforms a list of list sequences
    into a 3D Numpy array of shape `(num_samples, max_sent_len, max_word_len)`.

    Args:
        sequences: List of lists of lists.
        dtype: Type of the output sequences.

    # Returns
        x: Numpy array.
    """
    max_sent_len = 0
    max_word_len = 0
    for sent in sequences:
        max_sent_len = max(len(sent), max_sent_len)
        for word in sent:
            max_word_len = max(len(word), max_word_len)

    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        for j, word in enumerate(sent):
            x[i, j, : len(word)] = word

    return x
