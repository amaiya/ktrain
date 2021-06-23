Module ktrain.vision.models
===========================

Functions
---------

    
`build_cnn(num_classes, input_shape=(28, 28, 1), activation='softmax')`
:   

    
`build_predefined(name, num_classes, input_shape=(224, 224, 3), freeze_layers=None, activation='softmax', pt_fc=[], pt_ps=[])`
:   Builds a pre-defined architecture supported in Keras.
    
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

    
`build_visionmodel(name, num_classes, input_shape=(224, 224, 3), freeze_layers=2, activation='softmax', pt_fc=[], pt_ps=[])`
:   

    
`image_classifier(name, train_data, val_data=None, freeze_layers=None, metrics=['accuracy'], optimizer_name='adam', multilabel=None, pt_fc=[], pt_ps=[], verbose=1)`
:   Returns a pre-trained ResNet50 model ready to be fine-tuned
    for multi-class classification. By default, all layers are
    trainable/unfrozen.
    
    
    Args:
        name (string): one of {'pretrained_resnet50', 'resnet50', 'default_cnn'}
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

    
`image_model(name, train_data, val_data=None, freeze_layers=None, metrics=['accuracy'], optimizer_name='adam', multilabel=None, pt_fc=[], pt_ps=[], verbose=1)`
:   Returns a pre-trained ResNet50 model ready to be fine-tuned
    for multi-class classification or regression. By default, all layers are
    trainable/unfrozen.
    
    
    Args:
        name (string): one of {'pretrained_resnet50', 'resnet50', 'default_cnn'}
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

    
`image_regression_model(name, train_data, val_data=None, freeze_layers=None, metrics=['mae'], optimizer_name='adam', pt_fc=[], pt_ps=[], verbose=1)`
:   Returns a pre-trained ResNet50 model ready to be fine-tuned
    for multi-class classification. By default, all layers are
    trainable/unfrozen.
    
    
    Args:
        name (string): one of {'pretrained_resnet50', 'resnet50', 'default_cnn'}
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

    
`pretrained_datagen(data, name)`
:   

    
`print_image_classifiers()`
:   

    
`print_image_regression_models()`
: