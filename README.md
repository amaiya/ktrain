# ktrain

*ktrain* is a lightweight wrapper for the deep learning library [Keras](https://keras.io/) to help build, train, and deploy neural networks.  With only a few lines of code, ktrain allows you to easily and quickly:

- estimate an optimal learning rate for your model given your data using a Learning Rate Finder
- utilize learning rate schedules such as the [triangular policy](https://arxiv.org/abs/1506.01186), the [1cycle policy](https://arxiv.org/abs/1803.09820), and [SGDR](https://arxiv.org/abs/1608.03983) to effectively minimize loss and improve generalization
- employ fast and easy-to-use pre-canned models for both text classification (e.g., NBSVM, fastText, logreg) and image classification (e.g., ResNet, Wide ResNet, Inception)
- load and preprocess text and image data from a variety of formats 
- inspect data points that were misclassified to help improve your model
- leverage a simple prediction API for saving and deploying both models and data-preprocessing steps to make predictions on new raw data

#### Tutorial Notebooks
Please see the following tutorial notebooks for a guide on how to use *ktrain* on your projects:
* Tutorial 1:  [Introduction](https://github.com/amaiya/ktrain/blob/master/tutorial-01-introduction.ipynb)
* Tutorial 2:  [Tuning Learning Rates](https://github.com/amaiya/ktrain/blob/master/tutorial-02-tuning-learning-rates.ipynb)
* Tutorial 3: [Image Classification](https://github.com/amaiya/ktrain/blob/master/tutorial-03-image-classification.ipynb)
* Tutorial 4: [Text Classification](https://github.com/amaiya/ktrain/blob/master/tutorial-04-text-classification.ipynb)
* Tutorial A1: [Additional tricks](https://github.com/amaiya/ktrain/blob/master/tutorial-A1-additional-tricks.ipynb), which covers topics such as examining misclassifications, inspecting intermediate output of Keras models for debugging, and built-in callbacks.

Tasks such as text classification and image classification can be accomplished easily with 
only a few lines of code.

#### Example: Classifying Images of [Dogs and Cats](https://www.kaggle.com/c/dogs-vs-cats) Using *ktrain*
```
import ktrain
from ktrain import vision as vis

# load data
(train_data, val_data, preproc) = vis.images_from_folder(
                                              datadir='data/dogscats',
                                              data_aug = vis.get_data_aug(horizontal_flip=True),
                                              train_test_names=['train', 'valid'], 
                                              target_size=(224,224), color_mode='rgb')

# load model
model = vis.image_classifier('pretrained_resnet50', train_data, val_data, freeze_layers=80)

# wrap model and data in ktrain.Learner object
learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, 
                             workers=8, use_multiprocessing=False, batch_size=64)

# find good learning rate
learner.lr_find()             # briefly simulate training to find good learning rate
learner.lr_plot()             # visually identify best learning rate

# train using triangular policy with ModelCheckpoint and implicit ReduceLROnPlateau and EarlyStopping
learner.autofit(1e-4, checkpoint_folder='/tmp') 
```

### Installation

```
pip3 install ktrain
```



### Requirements

The following software/libraries should be installed:

- [Python 3.6+](https://www.python.org/) (tested on 3.6.7)
- [Keras](https://keras.io/)  (tested on 2.2.4)
- [TensorFlow](https://www.tensorflow.org/)  (tested on 1.10.1)
- [scikit-learn](https://scikit-learn.org/stable/) (tested on 0.20.0)
- [matplotlib](https://matplotlib.org/) (tested on 3.0.0)
- [pandas](https://pandas.pydata.org/) (tested on 0.24.2)
<!--- [PIL](https://pillow.readthedocs.io/en/stable/) (tested on 5.1.0) -->
<!-- - [scipy](https://www.scipy.org/) (tested on 1.2.1)  - insalled by scikit-learn-->
<!-- - [numpy](http://www.numpy.org/) (tested on 1.16.1)  - installed by scikit-learn -->

This code was tested on Ubuntu 18.04 LTS using Keras 2.2.4 with a TensorFlow 1.10 backend.
There are a few portions of the code that may explicitly depend on TensorFlow, but
such dependencies are kept to a minimum.

----
**Creator:  [Arun S. Maiya](http://arun.maiya.net)**

**Email:** arun [at] maiya [dot] net
