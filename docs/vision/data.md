Module ktrain.vision.data
=========================

Functions
---------

    
`datagen_needs_fit(datagen)`
:   

    
`detect_color_mode(train_directory, target_size=(32, 32))`
:   

    
`fit_datagens(train_datagen, test_datagen, train_array=None, train_directory=None, target_size=None, color_mode='rgb', flat_dir=False)`
:   computes stats of images for normalization

    
`get_data_aug(rotation_range=40, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=False, vertical_flip=False, featurewise_center=True, featurewise_std_normalization=True, samplewise_center=False, samplewise_std_normalization=False, rescale=None, **kwargs)`
:   This function is simply a wrapper around ImageDataGenerator
    with some reasonable defaults for data augmentation.
    Returns the default image_data_generator to support
    data augmentation and data normalization.
    Parameters can be adjusted by caller.
    Note that the ktrain.vision.model.image_classifier
    function may adjust these as needed.

    
`get_test_datagen(data_aug=None)`
:   

    
`images_from_array(x_train, y_train, validation_data=None, val_pct=0.1, random_state=None, data_aug=None, classes=None, class_names=None, is_regression=False)`
:   Returns image generator (Iterator instance) from training
    and validation data in the form of NumPy arrays.
    This function only supports image classification.
    For image regression, please use images_from_df.
    
    Args:
      x_train(numpy.ndarray):  training gdata
      y_train(numpy.ndarray):  labels must either be:
                               1. one-hot (or multi-hot) encoded arrays
                               2. integer values representing the label
      validation_data (tuple): tuple of numpy.ndarrays for validation data.
                               labels should be in one of the formats listed above.
      val_pct(float): percentage of training data to use for validaton if validation_data is None
      random_state(int): random state to use for splitting data
      data_aug(ImageDataGenerator):  a keras.preprocessing.image.ImageDataGenerator
      classes(str): old name for class_names - should no longer be used
      class_names(str): list of strings to use as class names
      is_regression(bool): If True, task is treated as regression. 
                           Used when there is single column of numeric values and
                           numeric values should be treated as numeric targets as opposed to class labels
    Returns:
      batches: a tuple of two image.Iterator - one for train and one for test and ImagePreprocessor instance

    
`images_from_csv(train_filepath, image_column, label_columns=[], directory=None, suffix='', val_filepath=None, is_regression=False, target_size=(224, 224), color_mode='rgb', data_aug=None, val_pct=0.1, random_state=None)`
:   Returns image generator (Iterator instance).
    Assumes output will be 2D one-hot-encoded labels for categorization.
    Note: This function preprocesses the input in preparation
          for a ResNet50 model.
    
    Args:
    train_filepath (string): path to training dataset in CSV format with header row
    image_column (string): name of column containing the filenames of images
                           If values in image_column do not have a file extension,
                           the extension should be supplied with suffix argument.
                           If values in image_column are not full file paths,
                           then the path to directory containing images should be supplied
                           as directory argument.
    
    label_columns(list or str): list or str representing the columns that store labels
                                Labels can be in any one of the following formats:
                                1. a single column string string (or integer) labels
    
                                   image_fname,label
                                   -----------------
                                   image01,cat
                                   image02,dog
    
                                2. multiple columns for one-hot-encoded labels
                                   image_fname,cat,dog
                                   image01,1,0
                                   image02,0,1
    
                                3. a single column of numeric values for image regression
                                   image_fname,age
                                   -----------------
                                   image01,68
                                   image02,18
    
    directory (string): path to directory containing images
                        not required if image_column contains full filepaths
    suffix(str): will be appended to each entry in image_column
                 Used when the filenames in image_column do not contain file extensions.
                 The extension in suffx should include ".".
    val_filepath (string): path to validation dataset in CSV format
    suffix(string): suffix to add to file names in image_column
    is_regression(bool): If True, task is treated as regression. 
                         Used when there is single column of numeric values and
                         numeric values should be treated as numeric targets as opposed to class labels
    target_size (tuple):  image dimensions
    color_mode (string):  color mode
    data_aug(ImageDataGenerator):  a keras.preprocessing.image.ImageDataGenerator
                                  for data augmentation
    val_pct(float):  proportion of training data to be used for validation
                     only used if val_filepath is None
    random_state(int): random seed for train/test split
    
    Returns:
    batches: a tuple of two Iterators - one for train and one for test

    
`images_from_df(train_df, image_column, label_columns=[], directory=None, val_directory=None, suffix='', val_df=None, is_regression=False, target_size=(224, 224), color_mode='rgb', data_aug=None, val_pct=0.1, random_state=None)`
:   Returns image generator (Iterator instance).
    Assumes output will be 2D one-hot-encoded labels for categorization.
    Note: This function preprocesses the input in preparation
          for a ResNet50 model.
    
    Args:
    train_df (DataFrame):  pandas dataframe for training dataset 
    image_column (string): name of column containing the filenames of images
                           If values in image_column do not have a file extension,
                           the extension should be supplied with suffix argument.
                           If values in image_column are not full file paths,
                           then the path to directory containing images should be supplied
                           as directory argument.
    
    label_columns(list or str): list or str representing the columns that store labels
                                Labels can be in any one of the following formats:
                                1. a single column string string (or integer) labels
    
                                   image_fname,label
                                   -----------------
                                   image01,cat
                                   image02,dog
    
                                2. multiple columns for one-hot-encoded labels
                                   image_fname,cat,dog
                                   image01,1,0
                                   image02,0,1
    
                                3. a single column of numeric values for image regression
                                   image_fname,age
                                   -----------------
                                   image01,68
                                   image02,18
    
    directory (string): path to directory containing images
                        not required if image_column contains full filepaths
    val_directory(strin): path to directory containing validation images.
                          only required if validation images are in different folder than train images
    suffix(str): will be appended to each entry in image_column
                 Used when the filenames in image_column do not contain file extensions.
                 The extension in suffx should include ".".
    val_df (DataFrame): pandas dataframe for validation set
    
    is_regression(bool): If True, task is treated as regression. 
                         Used when there is single column of numeric values and
                         numeric values should be treated as numeric targets as opposed to class labels
    target_size (tuple):  image dimensions
    color_mode (string):  color mode
    data_aug(ImageDataGenerator):  a keras.preprocessing.image.ImageDataGenerator
                                  for data augmentation
    val_pct(float):  proportion of training data to be used for validation
                     only used if val_filepath is None
    random_state(int): random seed for train/test split
    
    Returns:
    batches: a tuple of two Iterators - one for train and one for test

    
`images_from_fname(train_folder, pattern='([^/]+)_\\d+.jpg$', val_folder=None, is_regression=False, target_size=(224, 224), color_mode='rgb', data_aug=None, val_pct=0.1, random_state=None, verbose=1)`
:   Returns image generator (Iterator instance).
    
    Args:
    train_folder (str): directory containing images
    pat (str):  regular expression to extract class from file name of each image
                Example: r'([^/]+)_\d+.jpg$' to match 'english_setter' in 'english_setter_140.jpg'
                By default, it will extract classes from file names of the form:
                   <class_name>_<numbers>.jpg
    val_folder (str): directory containing validation images. default:None
    is_regression(bool): If True, task is treated as regression. 
                         Used when there is single column of numeric values and
                         numeric values should be treated as numeric targets as opposed to class labels
    target_size (tuple):  image dimensions
    color_mode (string):  color mode
    data_aug(ImageDataGenerator):  a keras.preprocessing.image.ImageDataGenerator
                                  for data augmentation
    val_pct(float):  proportion of training data to be used for validation
                     only used if val_folder is None
    random_state(int): random seed for train/test split
    verbose(bool):   verbosity
    
    Returns:
    batches: a tuple of two Iterators - one for train and one for test

    
`images_from_folder(datadir, target_size=(224, 224), classes=None, color_mode='rgb', train_test_names=['train', 'test'], data_aug=None, verbose=1)`
:   Returns image generator (Iterator instance).
    Assumes output will be 2D one-hot-encoded labels for categorization.
    Note: This function preprocesses the input in preparation
          for a ResNet50 model.
    
    Args:
    datadir (string): path to training (or validation/test) dataset
        Assumes folder follows this structure:
        ├── datadir
        │   ├── train
        │   │   ├── class0       # folder containing documents of class 0
        │   │   ├── class1       # folder containing documents of class 1
        │   │   ├── class2       # folder containing documents of class 2
        │   │   └── classN       # folder containing documents of class N
        │   └── test
        │       ├── class0       # folder containing documents of class 0
        │       ├── class1       # folder containing documents of class 1
        │       ├── class2       # folder containing documents of class 2
        │       └── classN       # folder containing documents of class N
    
    target_size (tuple):  image dimensions
    classes (list):  optional list of class subdirectories (e.g., ['cats','dogs'])
    color_mode (string):  color mode
    train_test_names(list): names for train and test subfolders
    data_aug(ImageDataGenerator):  a keras.preprocessing.image.ImageDataGenerator
                                  for data augmentation
    verbose (bool):               verbosity
    
    Returns:
    batches: a tuple of two Iterators - one for train and one for test

    
`preprocess_csv(csv_in, csv_out, x_col='filename', y_col=None, sep=',', label_sep=' ', suffix='', split_by=None)`
:   Takes a CSV where the one column contains a file name and a column
    containing a string representations of the class(es) like here:
    image_name,tags
    01, sunny|hot
    02, cloudy|cold
    03, cloudy|hot
    
    .... and one-hot encodes the classes to produce a CSV as follows:
    image_name, cloudy, cold, hot, sunny
    01.jpg,0,0,1,1
    02.jpg,1,1,0,0
    03.jpg,1,0,1,0
    Args:
        csv_in (str):  filepath to input CSV file
        csv_out (str): filepath to output CSV file
        x_col (str):  name of column containing file names
        y_col (str): name of column containing the classes
        sep (str): field delimiter of entire file (e.g., comma fore CSV)
        label_sep (str): delimiter for column containing classes
        suffix (str): adds suffix to x_col values
        split_by(str): name of column. A separate CSV will be
                       created for each value in column. Useful
                       for splitting a CSV based on whether a column
                       contains 'train' or 'valid'.
    Return:
        list :  the list of clases (and csv_out will be new CSV file)

    
`preview_data_aug(img_path, data_aug, rows=1, n=4)`
:   Preview data augmentation (ImageDatagenerator)
    on a supplied image.

    
`preview_data_aug_OLD(img_path, data_aug, n=4)`
:   Preview data augmentation (ImageDatagenerator)
    on a supplied image.

    
`process_datagen(data_aug, train_array=None, train_directory=None, target_size=None, color_mode='rgb', flat_dir=False)`
:   

    
`sample_image_folder(train_directory, target_size, color_mode='rgb', flat_dir=False)`
:   

    
`show_image(img_path)`
:   Given file path to image, show it in Jupyter notebook

    
`show_random_images(img_folder, n=4, rows=1)`
:   display random images from a img_folder