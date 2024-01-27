"""
Model definition.
"""

from .... import utils as U
from ....imports import *

# if U.is_tf_keras():
# from .layers import CRF
# else:
# from .layers_standalone import CRF


def save_model(model, weights_file, params_file):
    with open(params_file, "w") as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)
        model.save_weights(weights_file)


def load_model(weights_file, params_file):
    with open(params_file) as f:
        model = keras.models.model_from_json(f.read(), custom_objects={"CRF": CRF})
        model.load_weights(weights_file)

    return model


class BiLSTMCRF(object):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(
        self,
        num_labels,
        word_vocab_size,
        char_vocab_size=None,
        word_embedding_dim=100,
        char_embedding_dim=25,
        word_lstm_size=100,
        char_lstm_size=25,
        fc_dim=100,
        dropout=0.5,
        embeddings=None,
        use_char=True,
        use_crf=True,
        char_mask_zero=True,
        use_elmo=False,
        use_transformer_with_dim=None,
    ):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
            use_crf (boolean): use crf as last layer.
            char_mask_zero(boolean): mask zero for character embedding (see TF2 isse #33148 and #33069)
            use_elmo(boolean): If True, model will be configured to accept Elmo embeddings
                               as an additional input to word and character embeddings
            use_transformer_with_dim(int): If not None, model will be configured to accept
                                           transformer embeddings of given dimension
        """
        super(BiLSTMCRF).__init__()
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._use_char = use_char
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._num_labels = num_labels
        self._char_mask_zero = char_mask_zero
        self._use_elmo = use_elmo
        self._use_transformer_with_dim = use_transformer_with_dim

    def build(self):
        # build word embedding
        word_ids = keras.layers.Input(
            batch_shape=(None, None), dtype="int32", name="word_input"
        )
        inputs = [word_ids]
        embedding_list = []
        if self._embeddings is None:
            word_embeddings = keras.layers.Embedding(
                input_dim=self._word_vocab_size,
                output_dim=self._word_embedding_dim,
                mask_zero=True,
                name="word_embedding",
            )(word_ids)
        else:
            word_embeddings = keras.layers.Embedding(
                input_dim=self._embeddings.shape[0],
                output_dim=self._embeddings.shape[1],
                mask_zero=True,
                weights=[self._embeddings],
                name="word_embedding",
            )(word_ids)
        embedding_list.append(word_embeddings)

        # build character based word embedding
        if self._use_char:
            char_ids = keras.layers.Input(
                batch_shape=(None, None, None), dtype="int32", name="char_input"
            )
            inputs.append(char_ids)
            char_embeddings = keras.layers.Embedding(
                input_dim=self._char_vocab_size,
                output_dim=self._char_embedding_dim,
                mask_zero=self._char_mask_zero,
                name="char_embedding",
            )(char_ids)
            char_embeddings = keras.layers.TimeDistributed(
                keras.layers.Bidirectional(keras.layers.LSTM(self._char_lstm_size))
            )(char_embeddings)
            embedding_list.append(char_embeddings)

        # add elmo embedding
        if self._use_elmo:
            elmo_embeddings = keras.layers.Input(shape=(None, 1024), dtype="float32")
            inputs.append(elmo_embeddings)
            embedding_list.append(elmo_embeddings)

        # add transformer embedding
        if self._use_transformer_with_dim is not None:
            transformer_embeddings = keras.layers.Input(
                shape=(None, self._use_transformer_with_dim), dtype="float32"
            )
            inputs.append(transformer_embeddings)
            embedding_list.append(transformer_embeddings)

        # concatenate embeddings
        word_embeddings = (
            keras.layers.Concatenate()(embedding_list)
            if len(embedding_list) > 1
            else embedding_list[0]
        )

        # build model
        word_embeddings = keras.layers.Dropout(self._dropout)(word_embeddings)
        z = keras.layers.Bidirectional(
            keras.layers.LSTM(units=self._word_lstm_size, return_sequences=True)
        )(word_embeddings)
        z = keras.layers.Dense(self._fc_dim, activation="tanh")(z)

        if self._use_crf:
            from .layers import CRF

            crf = CRF(self._num_labels, sparse_target=False)
            loss = crf.loss_function
            pred = crf(z)
        else:
            loss = "categorical_crossentropy"
            pred = keras.layers.Dense(self._num_labels, activation="softmax")(z)

        model = keras.Model(inputs=inputs, outputs=pred)

        return model, loss
