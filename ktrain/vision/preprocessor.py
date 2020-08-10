from ..imports import *
from ..preprocessor import Preprocessor
from .. import utils as U
class ImagePreprocessor(Preprocessor):
    """
    Image preprocessing
    """

    def __init__(self, datagen, classes, target_size=(224,224), color_mode='rgb'):

        if not isinstance(datagen, image.ImageDataGenerator):
            raise ValueError('datagen must be instance of ImageDataGenerator')
        self.datagen = datagen
        self.c = classes
        self.target_size = target_size
        self.color_mode = color_mode


    def get_preprocessor(self):
        return self.datagen

    def get_classes(self):
        return self.c


    def preprocess_test(self, data, batch_size=U.DEFAULT_BS):
        """
        Alias for preprocess
        """
        return self.preprocess(data, batch_size=batch_size)


    def preprocess(self, data, batch_size=U.DEFAULT_BS):
        """
        Receives raw data and returns 
        tuple containing the generator and steps
        argument for model.predict.
        """
        # input is an array of pixel values
        if isinstance(data, np.ndarray):
            generator = self.datagen.flow(data, shuffle=False)
            generator.batch_size = batch_size
            nsamples = len(data)
            steps = math.ceil(nsamples/batch_size)
            return (generator, steps)

        # input is a folder of images
        elif os.path.isdir(data):
            folder = data
            if folder[-1] != os.sep: folder += os.sep
            parent = os.path.dirname(os.path.dirname(folder))
            folder_name = os.path.basename(os.path.dirname(folder))
            if self.target_size is None or self.color_mode is None:
                raise Exception('To use predict_folder, you must load the data using either '+\
                                'the images_from_folder function or the images_from_csv function.')
            generator= self.datagen.flow_from_directory(parent,
                                                       classes=[folder_name],
                                                       target_size=self.target_size,
                                                       class_mode='categorical',
                                                       shuffle=False,
                                                       interpolation='bicubic',
                                                       color_mode = self.color_mode)
            generator.batch_size = batch_size
            nsamples = generator.samples
            steps = math.ceil(nsamples/batch_size)
            return (generator, steps)
        # input is the path to an image file
        elif os.path.isfile(data):
            if self.target_size is None or self.color_mode is None:
                raise Exception('To use predict_filename, you must load the data using either '+\
                                'the ktrain.vision.images_from_folder function or the ' +\
                                'ktrain.vision.images_from_csv function.')
            img = image.load_img(data, target_size=self.target_size, color_mode=self.color_mode)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            generator =  self.datagen.flow(np.array(x), shuffle=False)
            generator.batch_size = batch_size
            nsamples = 1
            steps = math.ceil(nsamples/batch_size)
            return (generator, steps)
        else:
            raise ValueError('data argument is not valid file, folder, or numpy.ndarray')





