from .. import utils as U
from ..imports import *
from .wrn import create_wide_residual_network

PRETRAINED_RESNET50 = "pretrained_resnet50"
PRETRAINED_MOBILENET = "pretrained_mobilenet"
PRETRAINED_MOBILENETV3 = "pretrained_mobilenetv3"
PRETRAINED_INCEPTION = "pretrained_inception"
PRETRAINED_EFFICIENTNETB1 = "pretrained_efficientnetb1"
PRETRAINED_EFFICIENTNETB7 = "pretrained_efficientnetb7"
RESNET50 = "resnet50"
MOBILENET = "mobilenet"
MOBILENETV3 = "mobilenetv3"
INCEPTION = "inception"
EFFICIENTNETB1 = "efficientnetb1"
EFFICIENTNETB7 = "efficientnetb7"
CNN = "default_cnn"
WRN22 = "wrn22"
PRETRAINED_MODELS = [
    PRETRAINED_RESNET50,
    PRETRAINED_MOBILENET,
    PRETRAINED_MOBILENETV3,
    PRETRAINED_INCEPTION,
    PRETRAINED_EFFICIENTNETB1,
    PRETRAINED_EFFICIENTNETB7,
]
PREDEFINED_MODELS = PRETRAINED_MODELS + [
    RESNET50,
    MOBILENET,
    MOBILENETV3,
    INCEPTION,
    EFFICIENTNETB1,
    EFFICIENTNETB7,
]
IMAGE_CLASSIFIERS = {
    PRETRAINED_RESNET50: "50-layer Residual Network (pretrained on ImageNet)",
    RESNET50: "50-layer Resididual Network (randomly initialized) [https://arxiv.org/abs/1512.03385]",
    PRETRAINED_MOBILENET: "MobileNet Neural Network (pretrained on ImageNet)",
    MOBILENET: "MobileNet Neural Network (randomly initialized) [https://arxiv.org/abs/1704.04861]",
    PRETRAINED_MOBILENETV3: "MobileNetV3-Small Neural Network (pretrained on ImageNet)",
    MOBILENETV3: "MobileNetV3-Small Neural Network (randomly initialized) [https://arxiv.org/abs/1905.02244]",
    PRETRAINED_INCEPTION: "Inception Version 3  (pretrained on ImageNet)",
    INCEPTION: "Inception Version 3 (randomly initialized) [http://arxiv.org/abs/1512.00567]",
    PRETRAINED_EFFICIENTNETB1: "EfficientNet-B1 Neural Network (pretrained on ImageNet)",
    EFFICIENTNETB1: "EfficientNet-B1 Neural Network (pretrained on ImageNet) [https://arxiv.org/abs/1905.11946]",
    PRETRAINED_EFFICIENTNETB7: "EfficientNet-B7 Neural Network (pretrained on ImageNet)",
    EFFICIENTNETB7: "EfficientNet-B7 Neural Network (pretrained on ImageNet) [https://arxiv.org/abs/1905.11946]",
    WRN22: "22-layer Wide Residual Network (randomly initialized)",
    CNN: "a default LeNet-like Convolutional Neural Network",
}


def print_image_classifiers():
    for k, v in IMAGE_CLASSIFIERS.items():
        print("%s: %s" % (k, v))


def print_image_regression_models():
    for k, v in IMAGE_CLASSIFIERS.items():
        print("%s: %s" % (k, v))


def pretrained_datagen(data, name):
    if not data or not U.is_iter(data):
        return
    idg = data.image_data_generator
    if name == PRETRAINED_RESNET50:
        idg.preprocessing_function = keras.applications.resnet50.preprocess_input
        idg.ktrain_preproc = "resnet50"
        idg.rescale = None
        idg.featurewise_center = False
        idg.samplewise_center = False
        idg.featurewise_std_normalization = False
        idg.samplewise_std_normalization = False
        idg.zca_whitening = False
    elif name == PRETRAINED_MOBILENET:
        idg.preprocessing_function = keras.applications.mobilenet.preprocess_input
        idg.ktrain_preproc = "mobilenet"
        idg.rescale = None
        idg.featurewise_center = False
        idg.samplewise_center = False
        idg.featurewise_std_normalization = False
        idg.samplewise_std_normalization = False
        idg.zca_whitening = False
    elif name == PRETRAINED_MOBILENETV3:
        idg.preprocessing_function = keras.applications.mobilenet_v3.preprocess_input
        idg.ktrain_preproc = "mobilenetv3"
        idg.rescale = None
        idg.featurewise_center = False
        idg.samplewise_center = False
        idg.featurewise_std_normalization = False
        idg.samplewise_std_normalization = False
        idg.zca_whitening = False
    elif name == PRETRAINED_INCEPTION:
        idg.preprocessing_function = keras.applications.inception_v3.preprocess_input
        idg.ktrain_preproc = "inception"
        idg.rescale = None
        idg.featurewise_center = False
        idg.samplewise_center = False
        idg.featurewise_std_normalization = False
        idg.samplewise_std_normalization = False
        idg.zca_whitening = False
    elif name == PRETRAINED_EFFICIENTNETB1 or name == PRETRAINED_EFFICIENTNETB7:
        idg.preprocessing_function = keras.applications.efficientnet.preprocess_input
        idg.ktrain_preproc = "efficientnet"
        idg.rescale = None
        idg.featurewise_center = False
        idg.samplewise_center = False
        idg.featurewise_std_normalization = False
        idg.samplewise_std_normalization = False
        idg.zca_whitening = False

    return


def image_classifier(
    name,
    train_data,
    val_data=None,
    freeze_layers=None,
    metrics=None,
    optimizer_name=U.DEFAULT_OPT,
    multilabel=None,
    pt_fc=[],
    pt_ps=[],
    verbose=1,
):
    """
    ```
    Returns a pre-defined/pre-trained model ready to be trained/fine-tuned
    for multi-class classification. By default, all layers are
    trainable/unfrozen.


    Args:
        name (string): one of model shown on ktrain.vision.print_image_classifiers
        train_data (image.Iterator): train data. Note: Will be manipulated here!
        val_data (image.Iterator): validation data.  Note: Will be manipulated here!
        freeze_layers (int):  number of beginning layers to make untrainable
                            If None, then all layers except new Dense layers
                            will be frozen/untrainable.
        metrics (list):  metrics to use
        metrics(list): List of metrics to use.  If None: 'accuracy' is used for binar/multiclass,
                       'binary_accuracy' is used for multilabel classification, and 'mae' is used for regression
        optimizer_name(str|obj): name of Keras optimizer (e.g., 'adam', 'sgd') or instance of keras Optimizer
        multilabel(bool):  If True, model will be build to support
                           multilabel classification (labels are not mutually exclusive).
                           If False, binary/multiclassification model will be returned.
                           If None, multilabel status will be inferred from data.
        pt_fc (list of ints): number of hidden units in extra Dense layers
                                before final Dense layer of pretrained model.
                                Only takes effect if name in PRETRAINED_MODELS
        pt_ps (list of floats): dropout probabilities to use before
                                each extra Dense layer in pretrained model.
                                Only takes effect if name in PRETRAINED_MODELS
        verbose (int):         verbosity
    Return:
        model(Model):  the compiled model ready to be fine-tuned/trained

    ```
    """
    return image_model(
        name,
        train_data,
        val_data=val_data,
        freeze_layers=freeze_layers,
        metrics=metrics,
        optimizer_name=optimizer_name,
        multilabel=multilabel,
        pt_fc=pt_fc,
        pt_ps=pt_ps,
        verbose=verbose,
    )


def image_regression_model(
    name,
    train_data,
    val_data=None,
    freeze_layers=None,
    metrics=["mae"],
    optimizer_name=U.DEFAULT_OPT,
    pt_fc=[],
    pt_ps=[],
    verbose=1,
):
    """
    ```
    Returns a pre-defined/pre-trained model ready to be trained/fine-tuned
    for multi-class classification. By default, all layers are
    trainable/unfrozen.


    Args:
        name (string): one of model shown on ktrain.vision.print_image_regression_models
        train_data (image.Iterator): train data. Note: Will be manipulated here!
        val_data (image.Iterator): validation data.  Note: Will be manipulated here!
        freeze_layers (int):  number of beginning layers to make untrainable
                            If None, then all layers except new Dense layers
                            will be frozen/untrainable.
        metrics (list):  metrics to use
        optimizer_name(str): name of Keras optimizer (e.g., 'adam', 'sgd')
        multilabel(bool):  If True, model will be build to support
                           multilabel classificaiton (labels are not mutually exclusive).
                           If False, binary/multiclassification model will be returned.
                           If None, multilabel status will be inferred from data.
        pt_fc (list of ints): number of hidden units in extra Dense layers
                                before final Dense layer of pretrained model.
                                Only takes effect if name in PRETRAINED_MODELS
        pt_ps (list of floats): dropout probabilities to use before
                                each extra Dense layer in pretrained model.
                                Only takes effect if name in PRETRAINED_MODELS
        verbose (int):         verbosity
    Return:
        model(Model):  the compiled model ready to be fine-tuned/trained

    ```
    """

    return image_model(
        name,
        train_data,
        val_data=val_data,
        freeze_layers=freeze_layers,
        metrics=metrics,
        optimizer_name=optimizer_name,
        multilabel=False,
        pt_fc=pt_fc,
        pt_ps=pt_ps,
        verbose=verbose,
    )


def image_model(
    name,
    train_data,
    val_data=None,
    freeze_layers=None,
    metrics=None,
    optimizer_name=U.DEFAULT_OPT,
    multilabel=None,
    pt_fc=[],
    pt_ps=[],
    verbose=1,
):
    """
    ```
    Returns a pre-defined/pre-trained model ready to be trained/fine-tuned
    for multi-class classification or regression. By default, all layers are
    trainable/unfrozen.


    Args:
        name (string): one of model shown on ktrain.vision.print_image_classifiers
        train_data (image.Iterator): train data. Note: Will be manipulated here!
        val_data (image.Iterator): validation data.  Note: Will be manipulated here!
        freeze_layers (int):  number of beginning layers to make untrainable
                            If None, then all layers except new Dense layers
                            will be frozen/untrainable.
        metrics(list): List of metrics to use.  If None: 'accuracy' is used for binar/multiclass,
                       'binary_accuracy' is used for multilabel classification, and 'mae' is used for regression
        optimizer_name(str|obj): name of Keras optimizer (e.g., 'adam', 'sgd') or instance of Keras optimizer
        multilabel(bool):  If True, model will be build to support
                           multilabel classificaiton (labels are not mutually exclusive).
                           If False, binary/multiclassification model will be returned.
                           If None, multilabel status will be inferred from data.
        pt_fc (list of ints): number of hidden units in extra Dense layers
                                before final Dense layer of pretrained model.
                                Only takes effect if name in PRETRAINED_MODELS
        pt_ps (list of floats): dropout probabilities to use before
                                each extra Dense layer in pretrained model.
                                Only takes effect if name in PRETRAINED_MODELS
        verbose (int):         verbosity
    Return:
        model(Model):  the compiled model ready to be fine-tuned/trained

    ```
    """
    # arg check
    U.data_arg_check(train_data=train_data, train_required=True)
    if name not in list(IMAGE_CLASSIFIERS.keys()):
        raise ValueError("Unknown or unsupported model: %s" % (name))

    if not U.is_iter(train_data):
        raise ValueError(
            "train_data must be an Keras iterator "
            + "(e.g., DirectoryIterator, DataframIterator, "
            + "NumpyArrayIterator) - please use the ktrain.data.images_from* "
            + "functions"
        )

    # check for MobileNetV3
    if name in [PRETRAINED_MOBILENETV3, MOBILENETV3] and not HAS_MOBILENETV3:
        raise ValueError(
            f"You chose {name}, but it does not appear to be available in your version of TensorFlow."
        )

    # set pretrained flag
    pretrained = True if name in PRETRAINED_MODELS else False

    # adjust freeze_layers with warning
    if not pretrained and freeze_layers is not None and freeze_layers > 0:
        warnings.warn(
            "Only pretrained models (e.g., pretrained_resnet50) support freeze_layers. "
            + "Setting freeze_layers to 0. Use one of the following models if"
            + "desiring a model pretrained on ImageNet: %s" % (PRETRAINED_MODELS)
        )
        freeze_layers = 0

    if pretrained and val_data is None:
        raise ValueError(
            "val_data is required if selecting a pretrained model, "
            + "as normalization scheme will be altered."
        )

    # adjust the data augmentation based on model selected
    if pretrained:
        pretrained_datagen(train_data, name)
        pretrained_datagen(val_data, name)
        U.vprint(
            "The normalization scheme has been changed for use with a %s" % (name)
            + " model. If you decide to use a different model, please reload your"
            + " dataset with a ktrain.vision.data.images_from* function.\n",
            verbose=verbose,
        )

    # determine if multilabel
    if multilabel is None:
        multilabel = U.is_multilabel(train_data)
    is_regression = False
    if not multilabel and len(train_data[0][-1].shape) == 1:
        is_regression = True

    # set metrics
    if is_regression and metrics is None:
        metrics = ["mae"]
    elif multilabel and metrics is None:
        metrics = ["binary_accuracy"]
    elif metrics is None:
        metrics = ["accuracy"]

    # set loss and acivations
    loss_func = "categorical_crossentropy"
    activation = "softmax"
    if multilabel:
        loss_func = "binary_crossentropy"
        activation = "sigmoid"
    elif is_regression:
        loss_func = "mse"
        activation = None

    U.vprint("Is Multi-Label? %s" % (multilabel), verbose=verbose)
    U.vprint("Is Regression? %s" % (is_regression), verbose=verbose)

    # determine number of classes and shape
    num_classes = 1 if is_regression else U.nclasses_from_data(train_data)
    input_shape = U.shape_from_data(train_data)

    # ------------
    # build model
    # ------------
    model = build_visionmodel(
        name,
        num_classes,
        input_shape=input_shape,
        freeze_layers=freeze_layers,
        activation=activation,
        pt_fc=pt_fc,
        pt_ps=pt_ps,
    )
    model.compile(optimizer=optimizer_name, loss=loss_func, metrics=metrics)
    return model


def build_visionmodel(
    name,
    num_classes,
    input_shape=(224, 224, 3),
    freeze_layers=2,
    activation="softmax",
    pt_fc=[],
    pt_ps=[],
):
    if name in PREDEFINED_MODELS:
        model = build_predefined(
            name,
            num_classes,
            input_shape=input_shape,
            freeze_layers=freeze_layers,
            activation=activation,
            pt_fc=pt_fc,
            pt_ps=pt_ps,
        )
    elif name == CNN:
        model = build_cnn(num_classes, input_shape=input_shape, activation=activation)
    elif name == WRN22:
        model = create_wide_residual_network(
            input_shape,
            nb_classes=num_classes,
            N=3,
            k=6,
            dropout=0.00,
            activation=activation,
            verbose=0,
        )
    else:
        raise ValueError("Unknown model: %s" % (name))
    U.vprint("%s model created." % (name))

    return model


def build_cnn(num_classes, input_shape=(28, 28, 1), activation="softmax"):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=input_shape,
        )
    )
    model.add(
        keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal"
        )
    )
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.20))
    model.add(
        keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
    )
    model.add(
        keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(
        keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
    )
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(num_classes, activation=activation))
    return model


def build_predefined(
    name,
    num_classes,
    input_shape=(224, 224, 3),
    freeze_layers=None,
    activation="softmax",
    pt_fc=[],
    pt_ps=[],
):
    """
    ```
    Builds a pre-defined architecture supported in Keras.

    Args:
        name (str): one of ktrain.vision.model.PREDEFINED_MODELS
        num_classes (int): # of classes
        input_shape (tuple): the input shape including channels
        freeze_layers (int): number of early layers to freeze.
                             Only takes effect if name in PRETRAINED_MODELS.
                             If None and name in PRETRAINED_MODELS,
                             all layers except the "custom head"
                             fully-connected (Dense) layers are frozen.
        activation (str):    name of the Keras activation to use in final layer
        pt_fc (list of ints): number of hidden units in extra Dense layers
                                before final Dense layer of pretrained model
        pt_ps (list of floats): dropout probabilities to use before
                                      each extra Dense layer in pretrained model
    ```
    """

    # default parameters
    include_top = False
    input_tensor = None
    dropout = 0.5  # final dropout

    # setup pretrained
    weights = "imagenet" if name in PRETRAINED_MODELS else None

    # setup the pretrained network
    if name in [RESNET50, PRETRAINED_RESNET50]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = keras.applications.ResNet50(
                include_top=include_top,
                weights=weights,
                input_tensor=input_tensor,
                input_shape=input_shape,
            )
    elif name in [MOBILENET, PRETRAINED_MOBILENET]:
        net = keras.applications.MobileNet(
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
        )
    elif name in [MOBILENETV3, PRETRAINED_MOBILENETV3]:
        net = keras.applications.MobileNetV3Small(
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
        )
    elif name in [INCEPTION, PRETRAINED_INCEPTION]:
        net = keras.applications.InceptionV3(
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
        )
    elif name in [EFFICIENTNETB1, PRETRAINED_EFFICIENTNETB1]:
        net = keras.applications.EfficientNetB1(
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
        )
    elif name in [EFFICIENTNETB7, PRETRAINED_EFFICIENTNETB7]:
        net = keras.applications.EfficientNetB7(
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
        )
    else:
        raise ValueError("Unsupported model: %s" % (name))

    if freeze_layers is None:
        for layer in net.layers:
            layer.trainable = False

    x = net.output
    x = keras.layers.Flatten()(x)

    # xtra FCs in pretrained model
    if name in PRETRAINED_MODELS:
        if len(pt_fc) != len(pt_ps):
            raise ValueError("size off xtra_fc must match size of fc_dropouts")
        for i, fc in enumerate(pt_fc):
            p = pt_ps[i]
            fc_name = "fc%s" % (i)
            if p is not None:
                x = keras.layers.Dropout(p)(x)
            x = keras.layers.Dense(
                fc, activation="relu", kernel_initializer="he_normal", name=fc_name
            )(x)

    # final FC
    x = keras.layers.Dropout(dropout)(x)
    output_layer = keras.layers.Dense(
        num_classes, activation=activation, name=activation
    )(x)
    model = keras.Model(inputs=net.input, outputs=output_layer)

    if freeze_layers is not None:
        # set certain earlier layers as non-trainable
        for layer in model.layers[:freeze_layers]:
            layer.trainable = False
        for layer in model.layers[freeze_layers:]:
            layer.trainable = True

    # set optimizer, loss, and metrics and return model
    return model
