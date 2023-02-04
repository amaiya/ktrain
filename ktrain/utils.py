from .imports import *

# ------------------------------------------------------------------------------
# KTRAIN DEFAULTS
# ------------------------------------------------------------------------------
DEFAULT_WD = 0.01


def get_default_optimizer(lr=0.001, wd=DEFAULT_WD):
    from .lroptimize.optimization import AdamWeightDecay

    opt = AdamWeightDecay(
        learning_rate=lr,
        weight_decay_rate=wd,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["layer_norm", "bias"],
    )
    return opt


# Use vanilla Adam as default unless weight decay is explicitly set by user
# in which case AdamWeightDecay is default optimizer.
# See core.Learner.set_weight_decay for more information
# dep_fix
if "tensorflow" in sys.modules:
    DEFAULT_OPT = (
        "adam"
        if version.parse(tf.__version__) < version.parse("2.11")
        else tf.keras.optimizers.legacy.Adam()
    )
else:
    DEFAULT_OPT = "adam"
DEFAULT_BS = 32
DEFAULT_ES = 5
DEFAULT_ROP = 2
# from .lroptimize.optimization import AdamWeightDecay
# DEFAULT_OPT = AdamWeightDecay(learning_rate=0.001,
# weight_decay_rate=0.01,
# beta_1=0.9,
# beta_2=0.999,
# epsilon=1e-6,
# exclude_from_weight_decay=['layer_norm', 'bias'])
DEFAULT_TRANSFORMER_LAYERS = [-2]  # second-to-last hidden state
DEFAULT_TRANSFORMER_MAXLEN = 512
DEFAULT_TRANSFORMER_NUM_SPECIAL = 2
MODEL_BASENAME = "tf_model"
MODEL_NAME = MODEL_BASENAME + ".h5"
PREPROC_NAME = MODEL_BASENAME + ".preproc"


# ------------------------------------------------------------------------------
# DATA/MODEL INSPECTORS
# ------------------------------------------------------------------------------


def is_ktrain_dataset(data):
    from .dataset import Dataset

    return isinstance(data, Dataset)


def loss_fn_from_model(model):
    # dep_fix
    if version.parse(tf.__version__) < version.parse("2.2") or DISABLE_V2_BEHAVIOR:
        return model.loss_functions[0].fn
    else:  # TF 2.2.0
        return model.compiled_loss._get_loss_object(
            model.compiled_loss._losses[0].name
        ).fn


def metrics_from_model(model):
    msg = "Could not retrieve metrics list from compiled model"

    # dep_fix
    if version.parse(tf.__version__) < version.parse("2.2") or DISABLE_V2_BEHAVIOR:
        return model._compile_metrics
        # return [m.name for m in model.metrics] if is_tf_keras() else model.metrics
    else:  # TF >= 2.2.0
        mlist = model.compiled_metrics._metrics
        if isinstance(mlist, list) and isinstance(
            mlist[0], str
        ):  # metrics are strings prior to training
            return mlist
        elif isinstance(mlist, list) and isinstance(mlist[0], list):
            try:
                return [m.name for m in mlist[0]]
            except:
                warnings.warn(msg)
                return []
        elif isinstance(mlist, list) and hasattr(
            mlist[0], "name"
        ):  # tf.keras.metrics.AUC()
            try:
                return [m.name for m in mlist]
            except:
                warnings.warn(msg)
                return []

        else:
            warnings.warn(msg)
            return []


def is_classifier(model):
    """
    checks for classification and mutlilabel from model
    """
    is_classifier = False
    is_multilabel = False

    # get loss name
    loss = model.loss
    if callable(loss):
        if hasattr(loss, "__name__"):
            loss = loss.__name__
        elif hasattr(loss, "name"):
            loss = loss.name
        else:
            raise Exception("could not get loss name")

    # check for classification
    if loss in [
        "categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "binary_crossentropy",
    ]:
        is_classifier = True
    else:
        mlist = metrics_from_model(model)
        if isinstance(mlist, (list, np.ndarray)) and any(
            ["accuracy" in m for m in mlist]
        ):
            is_classifier = True
        elif isinstance(mlist, (list, np.ndarray)) and any(["auc" in m for m in mlist]):
            is_classifier = True

    # check for multilabel
    if loss == "binary_crossentropy":
        if is_huggingface(model=model):
            is_multilabel = True
        else:
            last = model.layers[-1]
            output_shape = last.output_shape
            mult_output = (
                True if len(output_shape) == 2 and output_shape[1] > 1 else False
            )
            if (
                (
                    hasattr(last, "activation")
                    and isinstance(last.activation, type(keras.activations.sigmoid))
                )
                or isinstance(last, type(keras.activations.sigmoid))
            ) and mult_output:
                is_multilabel = True
    return (is_classifier, is_multilabel)


def is_tabular_from_data(data):
    return type(data).__name__ in ["TabularDataset"]


def is_huggingface(model=None, data=None):
    """
    check for hugging face transformer model
    from  model and/or data
    """
    huggingface = False
    if model is not None and is_huggingface_from_model(model):
        huggingface = True
    elif data is not None and is_huggingface_from_data(data):
        huggingface = True
    return huggingface


def is_huggingface_from_model(model):
    # 20201202: support both transformers<4.0 and transformers>=4.0
    return "transformers.modeling_tf" in str(
        type(model)
    ) or "transformers.models" in str(type(model))


def is_huggingface_from_data(data):
    return type(data).__name__ in ["TransformerDataset"]


def is_ner(model=None, data=None):
    ner = False
    if data is None:
        warnings.warn("is_ner only detects CRF-based NER models when data is None")
    if model is not None and is_crf(model):
        ner = True
    elif data is not None and is_ner_from_data(data):
        ner = True
    return ner


def is_crf(model):
    """
    checks for CRF sequence tagger.
    """
    # loss = model.loss
    # if callable(loss):
    # if hasattr(loss, '__name__'):
    # loss = loss.__name__
    # elif hasattr(loss, 'name'):
    # loss = loss.name
    # else:
    # raise Exception('could not get loss name')
    # return loss == 'crf_loss' or 'CRF.loss_function' in str(model.loss)
    return type(model.layers[-1]).__name__ == "CRF"


# def is_ner_from_model(model):
# """
# checks for sequence tagger.
# Curently, only checks for a CRF-based sequence tagger
# """
# loss = model.loss
# if callable(loss):
# if hasattr(loss, '__name__'):
# loss = loss.__name__
# elif hasattr(loss, 'name'):
# loss = loss.name
# else:
# raise Exception('could not get loss name')

# return loss == 'crf_loss' or 'CRF.loss_function' in str(model.loss)


def is_ner_from_data(data):
    return type(data).__name__ == "NERSequence"


def is_nodeclass(model=None, data=None):
    result = False
    if data is not None and type(data).__name__ == "NodeSequenceWrapper":
        result = True
    return result


def is_linkpred(model=None, data=None):
    result = False
    if data is not None and type(data).__name__ == "LinkSequenceWrapper":
        result = True
    return result


def is_imageclass_from_data(data):
    return type(data).__name__ in [
        "DirectoryIterator",
        "DataFrameIterator",
        "NumpyArrayIterator",
    ]


def is_regression_from_data(data):
    """
    checks for regression task from data
    """
    data_arg_check(val_data=data, val_required=True)
    if is_ner(data=data):
        return False  # NERSequence
    elif is_nodeclass(data=data):
        return False  # NodeSequenceWrapper
    elif is_linkpred(data=data):
        return False  # LinkSequenceWrapper
    Y = y_from_data(data)
    if len(Y.shape) == 1 or (len(Y.shape) > 1 and Y.shape[1] == 1):
        return True
    return False


def is_multilabel(data):
    """
    checks for multilabel from data
    """
    data_arg_check(val_data=data, val_required=True)
    if is_ner(data=data):
        return False  # NERSequence
    elif is_nodeclass(data=data):
        return False  # NodeSequenceWrapper
    elif is_linkpred(data=data):
        return False  # LinkSequenceWrapper
    multilabel = False
    Y = y_from_data(data)
    if len(Y.shape) == 1 or (len(Y.shape) > 1 and Y.shape[1] == 1):
        return False
    for idx, y in enumerate(Y):
        if idx >= 1024:
            break
        if np.issubdtype(type(y), np.integer) or np.issubdtype(type(y), np.floating):
            return False
        total_for_example = sum(y)
        if total_for_example > 1:
            multilabel = True
            break
    return multilabel


def shape_from_data(data):
    err_msg = "could not determine shape from %s" % (type(data))
    if is_iter(data):
        if is_ktrain_dataset(data):
            return data.xshape()
        elif hasattr(data, "image_shape"):
            return data.image_shape  # DirectoryIterator/DataFrameIterator
        elif hasattr(data, "x"):  # NumpyIterator
            return data.x.shape[1:]
        else:
            try:
                return data[0][0].shape[1:]
            except:
                raise Exception(err_msg)
    else:
        try:
            if type(data[0]) == list:  # BERT-style tuple
                return data[0][0].shape
            else:
                return data[0].shape  # standard tuple
        except:
            raise Exception(err_msg)


def ondisk(data):
    if hasattr(data, "ondisk"):
        return data.ondisk()

    ondisk = is_iter(data) and (type(data).__name__ not in ["NumpyArrayIterator"])
    return ondisk


def nsamples_from_data(data):
    err_msg = "could not determine number of samples from %s" % (type(data))
    if is_iter(data):
        if is_ktrain_dataset(data):
            return data.nsamples()
        elif hasattr(data, "samples"):  # DirectoryIterator/DataFrameIterator
            return data.samples
        elif hasattr(data, "n"):  # DirectoryIterator/DataFrameIterator/NumpyIterator
            return data.n
        else:
            raise Exception(err_msg)
    else:
        try:
            if type(data[0]) == list:  # BERT-style tuple
                return len(data[0][0])
            else:
                return len(data[0])  # standard tuple
        except:
            raise Exception(err_msg)


def nclasses_from_data(data):
    if is_iter(data):
        if is_ktrain_dataset(data):
            return data.nclasses()
        elif hasattr(data, "classes"):  # DirectoryIterator
            return len(set(data.classes))
        else:
            try:
                return data[0][1].shape[1]  # DataFrameIterator/NumpyIterator
            except:
                raise Exception(
                    "could not determine number of classes from %s" % (type(data))
                )
    else:
        try:
            return data[1].shape[1]
        except:
            raise Exception(
                "could not determine number of classes from %s" % (type(data))
            )


def y_from_data(data):
    if is_iter(data):
        if is_ktrain_dataset(data):
            return data.get_y()
        elif hasattr(data, "classes"):  # DirectoryIterator
            return keras.utils.to_categorical(data.classes)
        elif hasattr(data, "labels"):  # DataFrameIterator
            return data.labels
        elif hasattr(data, "y"):  # NumpyArrayIterator
            # return to_categorical(data.y)
            return data.y
        else:
            raise Exception(
                "could not determine number of classes from %s" % (type(data))
            )
    else:
        try:
            return data[1]
        except:
            raise Exception(
                "could not determine number of classes from %s" % (type(data))
            )


def is_iter(data, ignore=False):
    if ignore:
        return True
    iter_classes = ["NumpyArrayIterator", "DirectoryIterator", "DataFrameIterator"]
    return data.__class__.__name__ in iter_classes or is_ktrain_dataset(data)


def data_arg_check(
    train_data=None,
    val_data=None,
    train_required=False,
    val_required=False,
    ndarray_only=False,
):
    if train_required and train_data is None:
        raise ValueError("train_data is required")
    if val_required and val_data is None:
        raise ValueError("val_data is required")
    if train_data is not None and not is_iter(train_data, ndarray_only):
        if bad_data_tuple(train_data):
            err_msg = "data must be tuple of numpy.ndarrays"
            if not ndarray_only:
                err_msg += " or an instance of ktrain.Dataset"
            raise ValueError(err_msg)
    if val_data is not None and not is_iter(val_data, ndarray_only):
        if bad_data_tuple(val_data):
            err_msg = "data must be tuple of numpy.ndarrays or BERT-style tuple"
            if not ndarray_only:
                err_msg += " or an instance of Iterator"
            raise ValueError(err_msg)
    return


def bert_data_tuple(data):
    """
    checks if data tuple is BERT-style format
    """
    if is_iter(data):
        return False
    if (
        type(data[0]) == list
        and len(data[0]) == 2
        and type(data[0][0]) is np.ndarray
        and type(data[0][1]) is np.ndarray
        and type(data[1]) is np.ndarray
        and np.count_nonzero(data[0][1]) == 0
    ):
        return True
    else:
        return False


def bad_data_tuple(data):
    """
    Checks for standard tuple or BERT-style tuple
    """
    if (
        not isinstance(data, tuple)
        or len(data) != 2
        or type(data[0]) not in [np.ndarray, list]
        or (type(data[0]) in [list] and type(data[0][0]) is not np.ndarray)
        or type(data[1]) is not np.ndarray
    ):
        return True
    else:
        return False


# ------------------------------------------------------------------------------
# PLOTTING UTILITIES
# ------------------------------------------------------------------------------


# plots images with labels within jupyter notebook
def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    # if type(ims[0]) is np.ndarray:
    # ims = np.array(ims).astype(np.uint8)
    # if (ims.shape[-1] != 3):
    # ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis("Off")
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else "none")


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ------------------------------------------------------------------------------
# DOWNLOAD UTILITIES
# ------------------------------------------------------------------------------


def download(url, filename):
    with open(filename, "wb") as f:
        response = requests.get(url, stream=True, verify=False)
        total = response.headers.get("content-length")

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            # print(total)
            for data in response.iter_content(
                chunk_size=max(int(total / 1000), 1024 * 1024)
            ):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write("\r[{}{}]".format("█" * done, "." * (50 - done)))
                sys.stdout.flush()


def get_ktrain_data():
    home = os.path.expanduser("~")
    ktrain_data = os.path.join(home, "ktrain_data")
    if not os.path.isdir(ktrain_data):
        os.mkdir(ktrain_data)
    return ktrain_data


# ------------------------------------------------------------------------------
# MISC UTILITIES
# ------------------------------------------------------------------------------

from subprocess import Popen


def checkjava(path=None):
    """
    Checks if a Java executable is available for Tika.
    Args:
        path(str): path to java executable
    Returns:
        True if Java is available, False otherwise
    """

    # Get path to java executable if path not set
    if not path:
        path = os.getenv("TIKA_JAVA", "java")

    # Check if java binary is available on path
    try:
        _ = Popen(path, stdout=open(os.devnull, "w"), stderr=open(os.devnull, "w"))
    except:
        return False
    return True


def batchify(X, size):
    """
    ```
    Splits X into separate batch sizes specified by size.
    Args:
        X(list): elements
        size(int): batch size
    Returns:
        list of evenly sized batches with the last batch having the remaining elements
    ```
    """
    return [X[x : x + size] for x in range(0, len(X), size)]


def list2chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def check_array(X, y=None, X_name="X", y_name="targets"):
    if not isinstance(X, (list, np.ndarray)):
        raise ValueError("%s must be a list or NumPy array" % X_name)
    if y is not None and not isinstance(y, (list, np.ndarray)):
        raise ValueError("%s must be a list or NumPy array" % y_name)
    return


def is_tf_keras():
    if keras.__name__ == "keras":
        is_tf_keras = False
    elif (
        keras.__name__
        in ["tensorflow.keras", "tensorflow.python.keras", "tensorflow_core.keras"]
        or keras.__version__[-3:] == "-tf"
    ):
        is_tf_keras = True
    else:
        raise KeyError("Cannot detect if using keras or tf.keras.")
    return is_tf_keras


def vprint(s=None, verbose=1):
    if not s:
        s = "\n"
    if verbose:
        print(s)


def add_headers_to_df(fname_in, header_dict, fname_out=None):
    df = pd.read_csv(fname_in, header=None)
    df.rename(columns=header_dict, inplace=True)
    if fname_out is None:
        name, ext = os.path.splitext(fname_in)
        name += "-headers"
        fname_out = name + "." + ext
    df.to_csv(fname_out, index=False)  # save to new csv file
    return


def get_random_colors(n, name="hsv", hex_format=True):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    cmap = plt.cm.get_cmap(name, n)
    result = []
    for i in range(n):
        color = cmap(i)
        if hex_format:
            color = rgb2hex(color)
        result.append(color)
    return np.array(result)


def get_hf_model_name(model_id):
    parts = model_id.split("/")
    if len(parts) == 1:
        model_id = parts[0]
    else:
        model_id = "/".join(parts[1:])
    if model_id.startswith("xlm-roberta"):
        model_name = "xlm-roberta"
    else:
        model_name = model_id.split("-")[0]
    return model_name


# ------------------------------------------------------------------------------
# target-handling
# ------------------------------------------------------------------------------
class YTransform:
    def __init__(self, class_names=[], label_encoder=None):
        """
        ```
        Cheks and transforms array of targets. Targets are transformed in place.
        Args:
          class_names(list):  labels associated with targets (e.g., ['negative', 'positive'])
                         Only used/required if:
                         1. targets are one/multi-hot-encoded
                         2. targets are integers and represent class IDs for classification task
                         Not required if:
                         1. targets are numeric and task is regression
                         2. targets are strings and task is classification (class_names are populated automatically)
          label_encoder(LabelEncoder): a prior instance of LabelEncoder.
                                       If None, will be created when train=True
        ```
        """
        if type(class_names) != list:
            if isinstance(class_names, (pd.Series, np.ndarray)):
                class_names = class_names.tolist()
            else:
                raise ValueError("class_names must be list")
        self.c = class_names
        self.le = label_encoder
        self.train_called = False

    def get_classes(self):
        return self.c

    def set_classes(self, class_names):
        self.c = (
            class_names.tolist() if isinstance(class_names, np.ndarray) else class_names
        )

    def apply(self, targets, train=True):
        if targets is None and train:
            raise ValueError("targets is None")
        elif targets is None and not train:
            return

        # validate labels against data
        targets = np.array(targets) if type(targets) == list else targets
        if len(targets.shape) > 1 and targets.shape[1] == 1:
            targets = np.squeeze(targets, axis=1)

        # handle numeric targets (regression)
        if len(targets.shape) == 1 and not isinstance(targets[0], str):
            # numeric targets
            if not self.get_classes() and train:
                warnings.warn(
                    "Task is being treated as REGRESSION because "
                    + "either class_names argument was not supplied or is_regression=True. "
                    + "If this is incorrect, change accordingly."
                )
            if not self.get_classes():
                targets = np.array(targets, dtype=np.float32)
        # string targets (classification)
        elif len(targets.shape) == 1 and isinstance(targets[0], str):
            if not train and self.le is None:
                raise ValueError(
                    "LabelEncoder has not been trained. Call with train=True"
                )
            if train:
                self.le = LabelEncoder()
                self.le.fit(targets)
                if self.get_classes():
                    warnings.warn(
                        "class_names argument was ignored, as they were extracted from string labels in dataset"
                    )
                self.set_classes(self.le.classes_)
            targets = self.le.transform(
                targets
            )  # convert to numerical targets for classfication
        # handle categorical targets (classification)
        elif len(targets.shape) > 1:
            if not self.get_classes():
                raise ValueError(
                    "targets are 1-hot or multi-hot encoded but class_names is empty. "
                    + "The classes argument should have been supplied."
                )
            else:
                if train and len(self.get_classes()) != targets.shape[1]:
                    raise ValueError(
                        "training targets suggest %s classes, but class_names are %s"
                        % (targets.shape[1], self.get_classes())
                    )

        # numeric targets (classification)
        if len(targets.shape) == 1 and self.get_classes():
            if np.issubdtype(type(max(targets)), np.floating):
                warnings.warn(
                    "class_names implies classification but targets array contains float(s) instead of integers or strings"
                )

            if train and (len(set(targets)) != int(max(targets) + 1)):
                raise ValueError(
                    "len(set(targets) is %s but max(targets)+1 is  %s"
                    % (len(set(targets)), int(max(targets) + 1))
                )
            targets = keras.utils.to_categorical(
                targets, num_classes=len(self.get_classes())
            )
        if train:
            self.train_called = True
        return targets

    def apply_train(self, targets):
        return self.apply(targets, train=True)

    def apply_test(self, targets):
        return self.apply(targets, train=False)


class YTransformDataFrame(YTransform):
    def __init__(self, label_columns=[], is_regression=False):
        """
        ```
        Checks and transforms label columns in DataFrame. DataFrame is modified in place
        Args:
          label_columns(list): list of columns storing labels
          is_regression(bool): If True, task is regression and integer targets are treated as numeric dependent variable.
                               IF False, task is classification and integer targets are treated as class IDs.
        ```
        """
        self.is_regression = is_regression
        if isinstance(label_columns, str):
            label_columns = [label_columns]
        self.label_columns = label_columns
        if not label_columns:
            raise ValueError("label_columns is required")
        self.label_columns = (
            [self.label_columns]
            if isinstance(self.label_columns, str)
            else self.label_columns
        )
        # class_names = label_columns if len(label_columns) > 1 else []
        super().__init__(class_names=[])

    def get_label_columns(self, squeeze=True):
        """
        Returns label columns of transformed DataFrame
        """
        if not self.train_called:
            raise Exception("apply_train should be called first")
        if not self.is_regression:
            new_lab_cols = self.c
        else:
            new_lab_cols = self.label_columns
        return new_lab_cols[0] if len(new_lab_cols) == 1 and squeeze else new_lab_cols

    def apply(self, df, train=True):
        df = (
            df.copy()
        )  # dep_fix: SettingWithCopy - prevent original DataFrame from losing old label columns

        labels_exist = True
        lst = self.label_columns[:]
        if not all(x in df.columns.values for x in lst):
            labels_exist = False
        if train and not labels_exist:
            raise ValueError(
                "dataframe is missing label columns: %s" % (self.label_columns)
            )

        # extract targets
        # todo: sort?
        if len(self.label_columns) > 1:
            if train and self.is_regression:
                warnings.warn(
                    "is_regression=True was supplied but ignored because multiple label columns imply classification"
                )
            cols = df.columns.values
            missing_cols = []
            for l in self.label_columns:
                if l not in df.columns.values:
                    missing_cols.append(l)
            if len(missing_cols) > 0:
                raise ValueError(
                    "These label_columns do not exist in df: %s" % (missing_cols)
                )

            # set targets
            targets = (
                df[self.label_columns].values
                if labels_exist
                else np.zeros((df.shape[0], len(self.label_columns)))
            )
            # set class names
            if train:
                self.set_classes(self.label_columns)
        # single column
        else:
            # set targets
            targets = (
                df[self.label_columns[0]].values
                if labels_exist
                else np.zeros(df.shape[0], dtype=np.int_)
            )
            if self.is_regression and isinstance(targets[0], str):
                warnings.warn(
                    "is_regression=True was supplied but targets are strings - casting to floats"
                )
                targets = targets.astype(np.float64)

            # set class_names if classification task and targets with integer labels
            if train and not self.is_regression and not isinstance(targets[0], str):
                class_names = list(set(targets))
                class_names.sort()
                class_names = list(map(str, class_names))
                if len(class_names) == 2:
                    class_names = [
                        "not_" + self.label_columns[0],
                        self.label_columns[0],
                    ]
                else:
                    class_names = [self.label_columns[0] + "_" + c for c in class_names]
                self.set_classes(class_names)

        # transform targets
        targets = super().apply(
            targets, train=train
        )  # self.c (new label_columns) may be modified here
        targets = (
            targets if len(targets.shape) > 1 else np.expand_dims(targets, 1)
        )  # since self.label_columns is list

        # modify DataFrame
        if labels_exist:
            for l in self.label_columns:
                del df[l]  # delete old label columns

        new_lab_cols = self.get_label_columns(squeeze=False)
        if len(new_lab_cols) != targets.shape[1]:
            raise ValueError(
                "mismatch between target shape and number of labels - please open ktrain GitHub issue"
            )
        for i, col in enumerate(new_lab_cols):
            df[col] = targets[:, i]
        df[new_lab_cols] = targets
        print(new_lab_cols)
        print(df[new_lab_cols].head())
        df[new_lab_cols] = df[new_lab_cols].astype("float32")

        return df

    def apply_train(self, df):
        return self.apply(df, train=True)

    def apply_test(self, df):
        return self.apply(df, train=False)
