from ..imports import *
from .. import utils as U
from .preprocessor import ImagePreprocessor


def show_image(img_path):
    """
    Given file path to image, show it in Jupyter notebook
    """
    if not os.path.isfile(img_path):
        raise ValueError('%s is not valid file' % (img_path))
    img = plt.imread(img_path)
    out = plt.imshow(img)
    return out


def show_random_images(img_folder, n=4, rows=1):
    """
    display random images from a img_folder
    """
    fnames = []
    for ext in ('*.gif', '*.png', '*.jpg'):
        fnames.extend(glob.glob(os.path.join(img_folder, ext)))
    ims = []
    for i in range(n):
        img_path = random.choice(fnames)
        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x = x/255.
        ims.append(x)
    U.plots(ims, rows=rows)
    return


def preview_data_aug(img_path, data_aug, rows=1, n=4):
    """
    Preview data augmentation (ImageDatagenerator)
    on a supplied image.
    """
    if type(img_path) != type('') or not os.path.isfile(img_path):
        raise ValueError('img_path must be valid file path to image')
    idg = copy.copy(data_aug)
    idg.featurewise_center = False
    idg.featurewise_std_normalization = False
    idg.samplewise_center = False
    idg.samplewise_std_normalization = False
    idg.rescale = None
    idg.zca_whitening = False
    idg.preprocessing_function = None

    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = x/255.
    x = x.reshape((1,) + x.shape)
    i = 0
    ims = []
    for batch in idg.flow(x, batch_size=1):
        ims.append(np.squeeze(batch))
        i += 1
        if i >= n: break
    U.plots(ims, rows=rows)
    return


def preview_data_aug_OLD(img_path, data_aug, n=4):
    """
    Preview data augmentation (ImageDatagenerator)
    on a supplied image.
    """
    if type(img_path) != type('') or not os.path.isfile(img_path):
        raise ValueError('img_path must be valid file path to image')
    idg = copy.copy(data_aug)
    idg.featurewise_center = False
    idg.featurewise_std_normalization = False
    idg.samplewise_center = False
    idg.samplewise_std_normalization = False
    idg.rescale = None
    idg.zca_whitening = False
    idg.preprocessing_function = None

    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = x/255.
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in idg.flow(x, batch_size=1):
        plt.figure()
        plt.imshow(np.squeeze(batch))
        i += 1
        if i >= n: break
    return



def get_data_aug(
                 rotation_range=40,
                 zoom_range=0.2,
                 width_shift_range=0.2,
                 height_shift_range=0.2,
                 horizontal_flip=False,
                 vertical_flip=False, 
                 featurewise_center=True,
                 featurewise_std_normalization=True,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 rescale=None,
                 **kwargs):
    """
    This function is simply a wrapper around ImageDataGenerator
    with some reasonable defaults for data augmentation.
    Returns the default image_data_generator to support
    data augmentation and data normalization.
    Parameters can be adjusted by caller.
    Note that the ktrain.vision.model.image_classifier
    function may adjust these as needed.
    """

    data_aug = image.ImageDataGenerator(
                                rotation_range=rotation_range,
                                zoom_range=zoom_range,
                                width_shift_range=width_shift_range,
                                height_shift_range=height_shift_range,
                                horizontal_flip=horizontal_flip,
                                vertical_flip=vertical_flip, 
                                featurewise_center=featurewise_center,
                                featurewise_std_normalization=featurewise_std_normalization,
                                samplewise_center=samplewise_center,
                                samplewise_std_normalization=samplewise_std_normalization,
                                rescale=rescale,
                                **kwargs)
    return data_aug


def get_test_datagen(data_aug=None):
    if data_aug:
        featurewise_center = data_aug.featurewise_center
        featurewise_std_normalization = data_aug.featurewise_std_normalization
        samplewise_center = data_aug.samplewise_center
        samplewise_std_normalization = data_aug.samplewise_std_normalization
        rescale = data_aug.rescale
        zca_whitening = data_aug.zca_whitening
        test_datagen = image.ImageDataGenerator(
                                rescale=rescale,
                                featurewise_center=featurewise_center,
                                samplewise_center=samplewise_center,
                                featurewise_std_normalization=featurewise_std_normalization,
                                samplewise_std_normalization=samplewise_std_normalization,
                                zca_whitening=zca_whitening)
    else:
        test_datagen = image.ImageDataGenerator()
    return test_datagen



def process_datagen(data_aug, train_array=None, train_directory=None,
                    target_size=None,
                    color_mode='rgb', 
                    flat_dir=False):
    # set generators for train and test
    if data_aug is not None:
        train_datagen = data_aug
        test_datagen = get_test_datagen(data_aug=data_aug)
    else:
        train_datagen = get_test_datagen()
        test_datagen = get_test_datagen()

    # compute statistics for normalization
    fit_datagens(train_datagen, test_datagen, 
                 train_array=train_array,
                 train_directory=train_directory,
                 target_size=target_size,
                 color_mode=color_mode, flat_dir=flat_dir)

    return (train_datagen, test_datagen)



def fit_datagens(train_datagen, test_datagen, 
                 train_array=None, train_directory=None,
                 target_size=None,
                 color_mode='rgb', flat_dir=False):
    """
    computes stats of images for normalization
    """
    if not datagen_needs_fit(train_datagen): return
    if bool(train_array is not None) == bool(train_directory):
        raise ValueError('only one of train_array or train_directory is required.')
    if train_array is not None:
        train_datagen.fit(train_array)
        test_datagen.fit(train_array)
    else:
        if target_size is None:
            raise ValueError('target_size is required when train_directory is supplied')
        fit_samples = sample_image_folder(train_directory, target_size, 
                                          color_mode=color_mode, flat_dir=flat_dir)
        train_datagen.fit(fit_samples)
        test_datagen.fit(fit_samples)
    return


def datagen_needs_fit(datagen):
    if datagen.featurewise_center or datagen.featurewise_std_normalization or \
       datagen.zca_whitening:
           return True
    else:
        return False

def sample_image_folder(train_directory, 
                         target_size,
                         color_mode='rgb', flat_dir=False):

    # adjust train_directory
    classes = None
    if flat_dir and train_directory is not None:
        folder = train_directory
        if folder[-1] != os.sep: folder += os.sep
        parent = os.path.dirname(os.path.dirname(folder))
        folder_name = os.path.basename(os.path.dirname(folder))
        train_directory = parent
        classes = [folder_name]

    # sample images
    batch_size = 100
    img_gen = image.ImageDataGenerator()
    batches = img_gen.flow_from_directory(
                directory=train_directory,
                classes=classes,
                target_size=target_size,
                batch_size=batch_size,
                color_mode=color_mode,
                shuffle=True)
    the_shape = batches[0][0].shape
    sample_size = the_shape[0]
    if K.image_data_format() == 'channels_first':
        num_channels = the_shape[1]
    else:
        num_channels = the_shape[-1]
    imgs, labels = next(batches)
    return imgs


def detect_color_mode(train_directory, 
                     target_size=(32,32)):
    try:
        fname = glob.glob(os.path.join(train_directory, '**/*'))[0]
        img = Image.open(fname).resize(target_size)
        num_channels = len(img.getbands())
        if num_channels == 3: return 'rgb'
        elif num_channels == 1: return 'grayscale'
        else: return 'rgby'
    except:
        warnings.warn('could not detect color_mode from %s' % (train_directory))
        return



def preprocess_csv(csv_in, csv_out, x_col='filename', y_col=None,
                   sep=',', label_sep=' ', suffix='', split_by=None):
    """
    Takes a CSV where the one column contains a file name and a column
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
    """
    if not y_col and not suffix:
        raise ValueError('one or both of y_col and suffix should be supplied')
    df = pd.read_csv(csv_in, sep=sep)
    f_csv_out = open(csv_out, 'w')
    writer = csv.writer(f_csv_out, delimiter=sep)
    if y_col: df[y_col] = df[y_col].apply(str)

    # write header
    if y_col:
        classes = set()
        for row in df.iterrows():
            data = row[1]
            tags = data[y_col].split(label_sep)
            classes.update(tags)
        classes = list(classes)
        classes.sort()
        writer.writerow([x_col] + classes)
    else:
        classes = df.columns[:-1]
        write.writerow(df.columns)

    # write rows
    for row in df.iterrows():
        data = row[1]
        data[x_col] = data[x_col] + suffix
        if y_col:
            out = list(data[[x_col]].values)
            tags = set(data[y_col].strip().split(label_sep))
            for c in classes:
                if c in tags: out.append(1)
                else: out.append(0)
        else:
            out = data
        writer.writerow(out)
    f_csv_out.close()
    return classes


def images_from_folder(datadir, target_size=(224,224),
                       classes=None,
                       color_mode='rgb',
                       train_test_names=['train', 'test'],
                       data_aug=None, verbose=1):

    """
    Returns image generator (Iterator instance).
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

    """

    # train/test names
    train_str = train_test_names[0]
    test_str = train_test_names[1]
    train_dir = os.path.join(datadir, train_str)
    test_dir = os.path.join(datadir, test_str)

    # color mode warning
    if PIL_INSTALLED:
        inferred_color_mode = detect_color_mode(train_dir)
        if inferred_color_mode is not None and (inferred_color_mode != color_mode):
            U.vprint('color_mode detected (%s) different than color_mode selected (%s)' % (inferred_color_mode, color_mode), 
                     verbose=verbose)

    # get train and test data generators
    (train_datagen, test_datagen) = process_datagen(data_aug, 
                                        train_directory=train_dir,
                                        target_size=target_size,
                                        color_mode=color_mode)
    batches_tr = train_datagen.flow_from_directory(train_dir,
                                         target_size=target_size,
                                         classes=classes,
                                         class_mode='categorical',
                                         shuffle=True,
                                         interpolation='bicubic',
                                         color_mode = color_mode)

    batches_te = test_datagen.flow_from_directory(test_dir,
                                              target_size=target_size,
                                              classes=classes,
                                              class_mode='categorical',
                                              shuffle=False,
                                              interpolation='bicubic',
                                              color_mode = color_mode)

    # setup preprocessor
    class_tup = sorted(batches_tr.class_indices.items(), key=operator.itemgetter(1))
    preproc = ImagePreprocessor(test_datagen, 
                                [x[0] for x in class_tup],
                                target_size=target_size, 
                                color_mode=color_mode)
    return (batches_tr, batches_te, preproc)


def images_from_csv(train_filepath, 
                   image_column,
                   label_columns=[],
                   directory=None,
                   suffix='',
                   val_filepath=None,
                   target_size=(224,224),
                    color_mode='rgb',
                    data_aug=None,
                    val_pct=0.1, random_state=None):

    """
    Returns image generator (Iterator instance).
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
                                1. a single column string string labels

                                   image_fname,label
                                   -----------------
                                   image01,cat
                                   image02,dog

                                2. multiple columns for one-hot-encoded labels
                                   image_fname,cat,dog
                                   image01,1,0
                                   image02,0,1

    directory (string): path to directory containing images
                        not required if image_column contains full filepaths
    suffix(str): will be appended to each entry in image_column
                 Used when the filenames in image_column do not contain file extensions.
                 The extension in suffx should include ".".
    val_filepath (string): path to validation dataset in CSV format
    suffix(string): suffix to add to file names in image_column 
    target_size (tuple):  image dimensions 
    color_mode (string):  color mode
    data_aug(ImageDataGenerator):  a keras.preprocessing.image.ImageDataGenerator
                                  for data augmentation
    val_pct(float):  proportion of training data to be used for validation
                     only used if val_filepath is None
    random_state(int): random seed for train/test split

    Returns:
    batches: a tuple of two Iterators - one for train and one for test

    """

    # get train and test data generators
    if directory:
        img_folder = directory
    else:
        df = pd.read_csv(train_filepath)
        img_folder =  os.path.dirname(df[image_column].iloc[0])
    (train_datagen, test_datagen) = process_datagen(data_aug, 
                                                    train_directory=img_folder,
                                                    target_size=target_size,
                                                    color_mode=color_mode,
                                                    flat_dir=True)

    # convert to dataframes
    df = pd.read_csv(train_filepath)
    if not val_filepath:
        if val_pct:
            prop = 1-val_pct
            if random_state is not None: np.random.seed(42)
            msk = np.random.rand(len(df)) < prop
            train_df = df[msk]
            val_df = df[~msk]
        else:
            val_df = None
    else:
        train_df = df
        val_df = pd.read_csv(val_filepath)

    # class names
    label_columns.sort()

    # fix file extensions, if necessary
    if suffix:
        train_df[image_column] = train_df[image_column].apply(lambda x : x+suffix)
        val_df[image_column] = val_df[image_column].apply(lambda x : x+suffix)


    # 1-hot-encode string labels
    if isinstance(label_columns, str) or \
       (isinstance(label_columns, (list, np.ndarray)) and len(label_columns) == 1):
        label_col_name = label_columns if isinstance(label_columns, str) else label_columns[0]
        if not isinstance(df[label_col_name].values[0], str):
            raise ValueError('If a single label column is provided, labels must be in the form of a string.')
        le = LabelEncoder()
        train_labels = train_df[label_col_name].values
        le.fit(train_labels)
        y_train = to_categorical(le.transform(train_labels))
        y_val = to_categorical(le.transform(val_df[label_col_name].values))
        y_train_pd = [y_train[:,i] for i in range(y_train.shape[1])]
        y_val_pd = [y_val[:,i] for i in range(y_val.shape[1])]
        label_columns = list(le.classes_)
        train_df = pd.DataFrame(zip(train_df[image_column].values, *y_train_pd), columns=[image_column]+label_columns)
        val_df = pd.DataFrame(zip(val_df[image_column].values, *y_val_pd), columns=[image_column]+label_columns)





    batches_tr = train_datagen.flow_from_dataframe(
                                      train_df,
                                      directory=directory,
                                      x_col = image_column,
                                      y_col=label_columns,
                                      target_size=target_size,
                                      class_mode='other',
                                      shuffle=True,
                                      interpolation='bicubic',
                                      color_mode = color_mode)

    batches_te = test_datagen.flow_from_dataframe(
                                      val_df,
                                      directory=directory,
                                      x_col = image_column,
                                      y_col=label_columns,
                                      target_size=target_size,
                                      class_mode='other',
                                      shuffle=False,
                                      interpolation='bicubic',
                                      color_mode = color_mode)

    # setup preprocessor 
    preproc = ImagePreprocessor(test_datagen, 
                                label_columns,
                                target_size=target_size, 
                                color_mode=color_mode)
    return (batches_tr, batches_te, preproc)



def images_from_fname( train_folder,
                      pattern=r'([^/]+)_\d+.jpg$',
                      val_folder=None,
                     target_size=(224,224),
                     color_mode='rgb',
                     data_aug=None,
                     val_pct=0.1, random_state=None,
                     verbose=1):

    """
    Returns image generator (Iterator instance).
	
    Args:
    train_folder (str): directory containing images
    pat (str):  regular expression to extract class from file name of each image
                Example: r'([^/]+)_\d+.jpg$' to match 'english_setter' in 'english_setter_140.jpg'
                By default, it will extract classes from file names of the form:
                   <class_name>_<numbers>.jpg
    val_folder (str): directory containing validation images. default:None
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

    """

    # get train and test data generators
    (train_datagen, test_datagen) = process_datagen(data_aug, 
                                                    train_directory=train_folder,
                                                    target_size=target_size,
                                                    color_mode=color_mode,
                                                    flat_dir=True)

    # train df
    train_df, class_names = _img_fnames_to_df(train_folder, pattern, verbose=verbose)

    # val df
    if val_folder is not None:
        val_df, _ = _img_fnames_to_df(val_folder, pattern, verbose=verbose)
    elif val_pct:
        prop = 1-val_pct
        if random_state is not None: np.random.seed(42)
        msk = np.random.rand(len(df)) < prop
        train_df = df[msk]
        val_df = df[~msk]
        val_folder = train_folder
    else:
        val_df = None

    batches_tr = train_datagen.flow_from_dataframe(
                                      train_df,
                                      directory=train_folder,
                                      x_col = 'image_name',
                                      y_col=class_names,
                                      target_size=target_size,
                                      class_mode='other',
                                      shuffle=True,
                                      interpolation='bicubic',
                                      color_mode = color_mode)

    if val_df is not None:
        batches_te = test_datagen.flow_from_dataframe(
                                          val_df,
                                          directory=val_folder,
                                          x_col = 'image_name',
                                          y_col=class_names,
                                          target_size=target_size,
                                          class_mode='other',
                                          shuffle=False,
                                          interpolation='bicubic',
                                          color_mode = color_mode)
    else:
        batches_te = None

    # setup preprocessor 
    preproc = ImagePreprocessor(test_datagen, 
                                class_names,
                                target_size=target_size, 
                                color_mode=color_mode)
    return (batches_tr, batches_te, preproc)



def _img_fnames_to_df(img_folder, pattern, verbose=1):
    # get fnames
    fnames = []
    for ext in ('*.gif', '*.png', '*.jpg'):
        fnames.extend(glob.glob(os.path.join(img_folder, ext)))

    # process filenames and labels
    image_names = []
    labels = []
    p = re.compile(pattern)
    for fname in fnames:
        r = p.search(fname)
        if r:
            image_names.append(os.path.basename(fname))
            labels.append(r.group(1))
        else:
            warnings.warn('Could not extract class for %s -  skipping this file'% (fname))
    class_names = list(set(labels))
    class_names.sort()
    c2i = {k:v for v,k in enumerate(class_names)}
    labels = [c2i[label] for label in labels]
    labels = to_categorical(labels)
    #class_names = [str(c) in class_names]
    U.vprint('Found %s classes: %s' % (len(class_names), class_names), verbose=verbose)
    U.vprint('y shape: (%s,%s)' % (labels.shape[0], labels.shape[1]), verbose=verbose)
    dct = {'image_name': image_names}
    for i in range(labels.shape[1]):
        dct[class_names[i]] = labels[:,i]

    # convert to dataframes
    df = pd.DataFrame(dct)
    return (df, class_names)



def images_from_array(x_train, y_train, 
                      validation_data=None,
                      data_aug=None):

    """
    Returns image generator (Iterator instance) from training
    and validation data in the form of NumPy arrays
    Assumes output will be 2D one-hot-encoded labels for categorization.
    Note: This function preprocesses the input in preparation
          for a ResNet50 model.
	
    Args:
    x_train(numpy.ndarray):  training gdata
    y_train(numpy.ndarray):  labels
                             Must be 1-hot encoded already.
    validation_data (tuple): tuple of numpy.ndarrays for validation data
    data_aug(ImageDataGenerator):  a keras.preprocessing.image.ImageDataGenerator
    Returns:
    batches: a tuple of two image.Iterator - one for train and one for test

    """

    # one-hot-encode if necessary
    if np.issubdtype(type(y_train[0]), np.integer) or\
        (isinstance(y_train[0], (list, np.ndarray)) and len(y_train[0]) == 1):
        y_train = to_categorical(y_train)
    if validation_data:
        x_test = validation_data[0]
        y_test = validation_data[1]
        if np.issubdtype(type(y_test[0]), np.integer) or\
           (isinstance(y_test[0], (list, np.ndarray)) and len(y_test[0]) == 1):
            y_test = to_categorical(y_test)


    (train_datagen, test_datagen) = process_datagen(data_aug, train_array=x_train)

    batches_tr = train_datagen.flow(x_train, y_train, shuffle=True)

    batches_te = None
    preproc = None
    if validation_data:
        batches_te = test_datagen.flow(x_test, y_test,
                                       shuffle=False)
        classes = map(str, list(range(len(y_train[0]))))
        preproc = ImagePreprocessor(test_datagen, classes, target_size=None, color_mode=None)
    return (batches_tr, batches_te, preproc)

