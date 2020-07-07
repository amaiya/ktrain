from .version import __version__
from . import imports as I
from .core import ArrayLearner, GenLearner, get_predictor, load_predictor, release_gpu_memory
from .vision.learner import ImageClassLearner
from .text.learner import BERTTextClassLearner, TransformerTextClassLearner
from .text.ner.learner import NERLearner
from .graph.learner import NodeClassLearner, LinkPredLearner
from .data import Dataset, TFDataset, SequenceDataset

from . import utils as U

__all__ = ['get_learner', 'get_predictor', 'load_predictor', 'release_gpu_memory',
           'Dataset', 'TFDataset', 'SequenceDataset']




def get_learner(model, train_data=None, val_data=None, 
                batch_size=U.DEFAULT_BS, eval_batch_size=U.DEFAULT_BS,
                workers=1, use_multiprocessing=False, multigpu=False):
    """
    Returns a Learner instance that can be used to tune and train Keras models.

    model (Model):        A compiled instance of keras.engine.training.Model
    train_data (tuple or generator): Either a: 
                                   1) tuple of (x_train, y_train), where x_train and 
                                      y_train are numpy.ndarrays or 
                                   2) Iterator
    val_data (tuple or generator): Either a: 
                                   1) tuple of (x_test, y_test), where x_testand 
                                      y_test are numpy.ndarrays or 
                                   2) Iterator
                                   Note: Should be same type as train_data.
    batch_size (int):              Batch size to use in training. default:32
    eval_batch_size(int):  batch size used by learner.predict
                           only applies to validaton data during training if
                           val_data is instance of utils.Sequence.
                           default:32
    workers (int): number of cpu processes used to load data.
                   This is ignored unless train_data/val_data is an instance of 
                   tf.keras.preprocessing.image.DirectoryIterator or tf.keras.preprocessing.image.DataFrameIterator. 
    use_multiprocessing(bool):  whether or not to use multiprocessing for workers
                               This is ignored unless train_data/val_data is an instance of 
                               tf.keras.preprocessing.image.DirectoryIterator or tf.keras.preprocessing.image.DataFrameIterator. 
    multigpu(bool):             Lets the Learner know that the model has been 
                                replicated on more than 1 GPU.
                                Only supported for models from vision.image_classifiers
                                at this time.
    """

    # check arguments
    if not isinstance(model, I.Model):
        raise ValueError('model must be of instance Model')
    U.data_arg_check(train_data=train_data, val_data=val_data)
    if type(workers) != type(1) or workers < 1:
        workers =1
    # check for NumpyArrayIterator 
    if train_data and not U.ondisk(train_data):
        if workers > 1 and not use_multiprocessing:
            use_multiprocessing = True
            wrn_msg = 'Changed use_multiprocessing to True because NumpyArrayIterator with workers>1'
            wrn_msg +=' is slow when use_multiprocessing=False.'
            wrn_msg += ' If you experience issues with this, please set workers=1 and use_multiprocessing=False.'
            I.warnings.warn(wrn_msg)

    # verify BERT
    is_bert = U.bert_data_tuple(train_data)
    if is_bert:
        maxlen = U.shape_from_data(train_data)[1]
        msg = """For a GPU with 12GB of RAM, the following maxima apply:
        sequence len=64, max_batch_size=64
        sequence len=128, max_batch_size=32
        sequence len=256, max_batch_size=16
        sequence len=320, max_batch_size=14
        sequence len=384, max_batch_size=12
        sequence len=512, max_batch_size=6
        
        You've exceeded these limits.
        If using a GPU with <=12GB of memory, you may run out of memory during training.
        If necessary, adjust sequence length or batch size based on above."""
        wrn = False
        if maxlen > 64 and batch_size > 64:
            wrn=True
        elif maxlen > 128 and batch_size>32:
            wrn=True
        elif maxlen>256 and batch_size>16:
            wrn=True
        elif maxlen>320 and batch_size>14:
            wrn=True
        elif maxlen>384 and batch_size>12:
            wrn=True
        elif maxlen > 512 and batch_size>6:
            wrn=True
        if wrn: I.warnings.warn(msg)


    # return the appropriate trainer
    if U.is_iter(train_data):
        if U.is_ner(model=model, data=train_data):
            learner = NERLearner
        elif U.is_imageclass_from_data(train_data):
            learner = ImageClassLearner
        elif U.is_nodeclass(data=train_data):
            learner = NodeClassLearner
        elif U.is_nodeclass(data=train_data):
            learner = LinkPredLearner
        elif U.is_huggingface(data=train_data):
            learner = TransformerTextClassLearner
        else:
            learner = GenLearner
    else:
        if is_bert: 
            learner = BERTTextClassLearner
        else: # vanilla text classifiers use standard ArrayLearners
            learner = ArrayLearner
    return learner(model, train_data=train_data, val_data=val_data, 
                   batch_size=batch_size, eval_batch_size=eval_batch_size, 
                   workers=workers, use_multiprocessing=use_multiprocessing, multigpu=multigpu)


# keys
# currently_unsupported: unsupported or disabled features (e.g., xai graph neural networks have not been implemented)
# dep_fix:  a fix to address a problem in a dependency
