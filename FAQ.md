# Frequently Asked Questions About *ktrain*

## Getting Started

- [I am a newcomer and am having trouble figuring out how to even get started. Where do I begin?](#i-am-a-newcomer-and-am-having-trouble-figuring-out-how-to-even-get-started-where-do-i-begin)

- [What kinds of applications have been built with *ktrain*?](#what-kinds-of-applications-have-been-built-with-ktrain)

- [How do I use ktrain with documents in PDF, DOC, or PPT formats?](#how-do-i-use-ktrain-with-documents-in-pdf-doc-or-ppt-formats)

- [Can I use ktrain without a GPU?](#can-i-use-ktrain-without-a-gpu)


## Installation/Deployment Issues
- [How do I install ktrain on a Windows machine?](#how-do-i-install-ktrain-on-a-windows-machine)

- [How do I use ktrain without an internet connection?](#how-do-i-use-ktrain-without-an-internet-connection)

- [Why am I seeing an ERROR when installing *ktrain* on Google Colab?](#why-am-i-seeing-an-error-when-installing-ktrain-on-google-colab)

- [Why does `texts_from_csv` throw an error on Google Cloud Storage?](#why-does-texts_from_csv-throw-an-error-on-google-cloud-storage)

- [How do I deploy a model using Flask?](#how-do-i-deploy-a-model-using-flask)

- [Why am I getting a 404 client error?](#why-am-i-getting-a-404-client-error)

- [How do I convert a model to ONNX for deployment?](#how-do-i-make-quantized-predictions-with-transformers-models)


## Training


- [How do I resume training from a saved checkpoint?](#how-do-i-resume-training-from-a-saved-checkpoint)

- [How do I save and/or reload a trained model?](#how-do-i-resume-training-from-a-saved-checkpoint)

- [How do I train using multiple GPUs?](#how-do-i-train-using-multiple-gpus)

- [How do I train a model using mixed precision?](#how-do-i-train-a-model-using-mixed-precision)

- [How do I handle imbalanced datasets?](#how-do-i-handle-imbalanced-datasets)

- [How do I use custom loss functions or optimizers?](#how-do-i-use-custom-loss-functions-or-optimizers)

- [How do I retrieve or visualize training history?](#how-do-i-retrieve-or-visualize-training-history)

- [I have a model that accepts multiple inputs (e.g., both text and other numerical or categorical variables).  How do I train it with *ktrain*?](#i-have-a-model-that-accepts-multiple-inputs-eg-both-text-and-other-numerical-or-categorical-variables--how-do-i-train-it-with-ktrain)

- [Can I use `tf.data.Dataset` instances with *ktrain*?](#can-i-use-tfdatadataset-instances-with-ktrain)

- [Why am I seeing a "list index out of range" error when calling predict?](#why-am-i-seeing-a-list-index-out-of-range-error-when-calling-predict)

- [How do I train a transformers model from a saved checkpoint folder?](#how-do-i-train-a-transformers-model-from-a-saved-checkpoint-folder)

- [How do pretrain a language model for use with ktrain?](#how-do-i-pretrain-a-language-model-for-use-with-ktrain)

- [How do I get reproducible results?](#how-do-i-get-reproducible-results)




## Evaluation, Inspection, and Prediction
- [How do I get the predicted class "probabilities" of a model?](#how-do-i-get-the-predicted-class-probabilities-of-a-model)

- [How do I use custom metrics with ktrain?](#how-do-i-use-custom-metrics-with-ktrain)

- [How do I obtain the word or sentence embeddings after fine-tuning a Transformer-based text classifier?](#how-do-i-obtain-the-word-or-sentence-embeddings-after-fine-tuning-a-transformer-based-text-classifier)

- [Running `predictor.explain` for text classification is slow.  How can I speed it up?](#running-predictorexplain-for-text-classification-is-slow--how-can-i-speed-it-up)

- [Running `preprocess_train` for Transformer models is slow.  How can I speed it up?](#running-preprocess_train-for-transformer-models-is-slow--how-can-i-speed-it-up)

- [How do I make quantized predictions with `transformers` models?](#how-do-i-make-quantized-predictions-with-transformers-models)

- [How do I increase batch size for predictions?](#how-do-i-increase-batch-size-for-predictions)

- [How do I speed up predictions?](#how-do-i-increase-batch-size-for-predictions)

- [How do I do cross validation with transformers?](#how-do-i-do-cross-validation-with-transformers)



---






### I am a newcomer and am having trouble figuring out how to even get started. Where do I begin?

Machine learning models (e.g., neural networks) are trained on example inputs and outputs to learn mappings between them.  Once trained, given a new input, a correct output can be predicted.
For example, if you train a neural network on documents as inputs and document categories (e.g., subject areas) as outputs, the neural network will learn to predict the categories of new documents.  


Training neural network models can be computationally intensive due to the number of mathematical operations it takes to learn the mappings.  GPUs (or Graphical Processing Units) are devices
that allow you train neural networks faster by performing many mathematical operations at the same time. 

*ktrain* is a Python library that allows you train a neural network and make predictions using a minimal number of "commands" or lines of code. 
It is built on top of a library by Google called TensorFlow. Only very basic and minimal Python knowledge is required to use it.  


A challenge for newcomers is setting up the programming environment.  This includes 1) gaining access to a computer with a GPU, 2) installing and setting up the TensorFlow library to use the GPU, and 3) setting up Jupyter notebook.
(A Jupyter notebook is a programming environment that allows you to type code and see and save results of that code in an interacive fashion.)  
Fortunately, Google did a nice thing and made notebook environments with GPU access freely available "in the cloud" to anyone with a Gmail account.

Here is how you can quickly get started using *ktrain*:

1. Go to [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) and sign in using your Gmail account.
2. Go to this [example notebook on image classification](https://colab.research.google.com/drive/1WipQJUPL7zqyvLT10yekxf_HNMXDDtyR). 
3. Save the notebook to your Google Drive: `File --> Save a copy in Drive`
4. Make sure the notebook is setup to use a GPU: `Runtime --> Change runtime type` and select `GPU` in the menu.
5. Click on each cell in the notebook and execute it by pressing `SHIFT` and `ENTER` at the same time. The notebook shows you how to build a neural network that recoginizes cats vs. dogs in photos.

If you're on a Windows laptop, you can [follow these Windows installation instructions for TensorFlow and ktrain](#how-do-i-install-ktrain-on-a-windows-machine) and try out *ktrain* locally.

Next, you can go through [the tutorials](https://github.com/amaiya/ktrain#tutorials) to learn more.  If you have questions about a method or function, 
type a question mark before the method and press ENTER in a Google Colab or Jupyter notebook to learn more.  Example: `?learner.autofit`.

- For more information on Python, see [here](https://learnpythonthehardway.org/).

- For more information on neural networks, see [this page](https://victorzhou.com/blog/intro-to-neural-networks/).

- For more information on Google Colab, see [this video](https://www.youtube.com/watch?v=inN8seMm7UI).

- For more information on Jupyter notebooks, see [this video](https://www.youtube.com/watch?v=HW29067qVWk).

*ktrain* is inspired by some other libraries like `fastai` and `ludwig`. For a deeper dive into neural networks, the *fastai MOOC* and the
*TensorFlow and Deep Learning Without a PhD* series are recommended.


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I resume training from a saved checkpoint?

This answer shows different ways to save/reload a model and resume training.

#### Method 1: Using Predictor API (RECOMMENDED - works for any model)
```python
# save Predictor (i.e., model and Preprocessor instance) after partially training
ktrain.get_predictor(model, preproc).save('/tmp/my_predictor')

# reload Predictor and extract model
model = ktrain.load_predictor('/tmp/my_predictor').model

# re-instantiate Learner and continue training
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=8)
learner.fit_onecycle(2e-5, 1)
```
Note that `preproc` here is a *Preprocessor* instance.  If using a data-loading function like `texts_from_csv` or `images_from_folder`, it will be the third return value from the function. Or, if using the [Transformer API](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A3-hugging_face_transformers.ipynb) for text classification, it will be the output of invoking `text.Transformer` (i.e., `preproc = text.Transformer('bert-base-uncased', ...)`).  Also, `trn` and `val` are typically the result of invoking `preproc.preprocess_train` and `preproc.preprocess_test`, respectively.


#### Method 2: Using `transformers` library (if training Hugging Face Transformers model)
If the model is a Hugging Face transformers model, you can use `transformers` directly:
```python
# save model using transformers API after partially training
learner.model.save_pretrained('/tmp/my_model')

# reload the model using transformers directly
from transformers import *
model = TFAutoModelForSequenceClassification.from_pretrained('/tmp/my_model')
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

# re-instantiate Learner and continue training
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=8)
learner.fit_onecycle(2e-5, 1)
```
**Note:**  You may need to [supply the number of classes](https://stackoverflow.com/a/62328920/13550699) as an argument to `TFAutoModelForSequenceClassification.from_pretrained`.  See the [transformers documentation](https://huggingface.co/transformers/quicktour.html) for more detail.  **Method 1** does this automatically for you.

#### Method 3: Using `checkpoint_folder` argument to save model weights

The `checkpoint_folder` argument (e.g., `learner.autofit(1e-4, 4, checkpoint_folder='/tmp/saved_weights')`), saves the weights only of the model after each epoch. 
The weights of any epoch can be reloaded into the model using the `model.load_weights` method as you normally would in `tf.Keras`.  You just need to first re-create
the model.  For instance, if training an NER model, it would work as follows:
```python
# recreate model from scratch
import ktrain
from ktrain import text
model = text.sequence_tagger(...
# load checkpoint weights into model
model.load_weights('../models/checkpoints/weights-10.hdf5')
# recreate learner
learner = ktrain.get_learner(model, ...
# continue training here
```

Finally, there is also a `learner.save_model` and `learner.load_model` methods intended for saving and reloading models when training interactively during a single session.



[[Back to Top](#frequently-asked-questions-about-ktrain)]

### How do I obtain the word or sentence embeddings after fine-tuning a Transformer-based text classifier?
Here is a self-contained example of generating word embeddings from a fine-tuned `Transformer` model:

```python
# load text data
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
train_b = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
test_b = fetch_20newsgroups(subset='test',categories=categories, shuffle=True)
(x_train, y_train) = (train_b.data, train_b.target)
(x_test, y_test) = (test_b.data, test_b.target)

# build, train, and validate model (Transformer is wrapper around transformers library)
import ktrain
from ktrain import text
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=train_b.target_names)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
learner.fit_onecycle(5e-5, 1)

# load model to generate embeddings
learner.model.save_pretrained('/tmp/mymodel')
from transformers import *
import tensorflow as tf
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModel.from_pretrained('/tmp/mymodel')
input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states.numpy().shape)  # print shape of embedding vectors
```
This will produce a vector for each word (and subword) in the input string.  For sentence embeddings, you can aggregate
in various ways (e.g., average vectors).

See also [this post](https://github.com/huggingface/transformers/issues/1950) on the `transformers` GitHub repo.

Note that, once a `transformers` model is trained and saved (e.g., using `predictor.save` or `learner.save_model` or `learner.model.save_pretrained`), it 
can be reloaded into other libraries that support `transformers` (e.g., `sentence-transformers`).

[[Back to Top](#frequently-asked-questions-about-ktrain)]



### How do I install ktrain on a Windows machine?

Here are detailed instructions for getting started with *ktrain* and TensorFlow on a Windows 10 computer.

#### Installation on Windows

1. Download and Install the [Miniconda Python distribution](https://docs.conda.io/en/latest/miniconda.html).  You will most likely want the **Python 3.8 Miniconda3 Windows 64-bit**.
2. Download and Install the [Microsft Visual C++ Redistributable](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)
3. Click on **Anaconda Powershell Prompt** in the Start Menu.
4. Create a conda environment for *ktrain*: `conda create -n kt python=3.7; conda activate kt`
5. Type: `pip install -U pip setuptools_scm jupyter` (run twice if error or use `--user` option)
6. [Install TensorFlow 2](https://www.tensorflow.org/install): `pip install tensorflow==2.3`
6. Type: `pip install ktrain`

If your machine has a GPU (which is needed for larger models), you'll need to perform [GPU setup for TensorFlow](https://www.tensorflow.org/install/gpu).

#### Resolving Problems
- If you experience a **Kernel Error** when running `jupyter notebook`, follow the [instructions here](https://stackoverflow.com/a/60611014)
  and copy the two files in `C:\Users\<your_user_name>\Miniconda3\envs\kt\Lib\site-packages\pywin32_system32` to `C:\Windows\System32`.
- If you experience SSL certificate problems with either `pip` or `conda`, run `conda config --set ssl_verify false` and 
replace all `pip` comands above with `pip --trusted-host pypi.org --trusted-host files.pythonhosted.org`.
- In the instructions above, we are installing TensorFlow 2.3. Note that there is a bug in both TensorFlow 2.3 and 2.2 affecting the *Learning-Rate-Finder* [that will not be fixed until TensorFlow 2.4](https://github.com/tensorflow/tensorflow/issues/41174#issuecomment-656330268).  The bug causes the learning-rate-finder to complete all epochs even after loss has diverged (i.e., no automatic-stopping).
- If using `tensorflow<=2.1`, you must also downgrade **transformers** to `transformers==3.1` to avoid errors.
- We have selected Python 3.7 in STEP 4 above with `python=3.7` for illustration purposes, but Python 3.8 is default if removed. 

#### Running an Example
Once installed, you can fire up Jupyter notebook (type:`jupyter notebook` at command prompt) and test out *ktrain* with something like this:
```python
# download Cats vs. Dogs image classification dataset
!curl -k --output C:/temp/cats_and_dogs_filtered.zip --url https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip 
import os
import zipfile
local_zip = 'C:/temp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:/temp')
zip_ref.close()

# train model
import ktrain
from ktrain import vision as vis
(trn, val, preproc) = vis.images_from_folder(
                                              datadir='C:/temp/cats_and_dogs_filtered',
                                              data_aug = vis.get_data_aug(horizontal_flip=True),
                                              train_test_names=['train', 'validation'])
learner = ktrain.get_learner(model=vis.image_classifier('pretrained_mobilenet', trn, val, freeze_layers=15), 
                             train_data=trn, val_data=val, workers=4, batch_size=64)
learner.fit_onecycle(1e-4, 1)

# make prediction
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.predict_filename('C:/temp/cats_and_dogs_filtered/validation/cats/cat.2000.jpg')
```




[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I use ktrain without an internet connection?

When using pretrained models or pretrained word embeddings in *ktrain*, files are automatically downloaded.  For instance,
pretrained models and vocabulary files from the `transformers` library are downloaded to `<home_directory>/.cache/huggingface/transformers` (or `<home_directory>/.cache/torch/transformers` in older versions)
by default.  Other data like pretrained word vectors are downloaded to the `<home_directory>/ktrain_data` folder.

In some settings, it is necessary to either train models or make predictions in environments with no internet 
access (e.g., behind a firewall, air-gapped networks).  Typically, it is sufficient to copy the above folders
to the machine without internet access. For instance, if loading and using a `Predictor` instance associated with a `transformers` model as shown below,
then all that is typically needed is a vocabulary file that is typically retrieved from the cache:
```python
# this should work on machine with no internet connectivity if cache folder is populated correctly
p = ktrain.load_predictor('/tmp/mypred')
p.predict(data)
```

In some cases (e.g., when training a model on a system with no internet access or using pretrained model for question-answering),
 due to a [current bug](https://github.com/huggingface/transformers/issues/5016) in the `transformers` library, files from `<home_directory>/.cache/torch/transformers` may
not load when there is no internet access even when present in the cache.  To get around this, you can download the model files to a folder and point
*ktrain* to the folder.  There are typically three files you need, and it is important that the downloaded files are rennamed 
to `tf_model.h5`, `config.json`, and `vocab.txt`.  We will show two examples of training and/or applying Hugging Face `transformers` models
**without** an internet connection.

#### Example 1: Text Classification (with no internet)
1. Download the model files.  There are two different ways to do this:
  - **Method 1:** On a machine with public internet access, go to the Hugging Face model repository: [https://huggingface.co/models](https://huggingface.co/models), 
    click on "List all files in model", and download `tf_model.h5`, `config.json`, and `vocab.txt`. It is important that these downloaded files are renamed specifically 
    to the three aforementioned file names. If you do not see a link to one or more of the required files (e.g., `vocab.txt` is sometimes not listed), you will have to
    download it using **Method 2**.
  - **Method 2:** 

    1. Make sure  cache folder, `<home_directory>/.cache/torch/transformers`, is empty.  
    2. On a machine with public internet access, run the following steps to download the model files to the cache folder (replace `MODEL_NAME` with model you want):
     ```python
	from ktrain import text
	MODEL_NAME = 'distilbert-base-uncased'
	dummy_texts = ['hello world', 'goodbye world', 'hi world']
	dummy_labels = ['hello', 'bye', 'hello']
	t = text.Transformer(MODEL_NAME, maxlen=500)
	trn = t.preprocess_train(dummy_texts, dummy_labels)
	model = t.get_classifier()
    ```
    3. After the previous step, the cache folder will contain the three required files, but these files will be named with random characters. Each of
       the model files has a corresponding `.json` file that contains the URL from where the model file was downloaded. On a Linux machine,
       you can type `grep etag *.json` to see which file names map to which required file:
       ```
       $ grep  etag *.json
        26bc1ad6.542ce428.json:{"url": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt", "etag": "\"64800d5d8528ce344256daf115d4965e\""}
        a41e817d.8949e27a.json:{"url": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-config.json", "etag": "\"73e3e66b2b29478be775da997515e69a\""}
        cce28882.e02bd57e.h5.json:{"url": "https://cdn.huggingface.co/distilbert-base-uncased-tf_model.h5", "etag": "\"b02023739d9f6377fc63d88926b29118-44\""}
       ```
       In the example above, you would rename `26bc1ad6.542ce428` to `vocab.txt`, rename` a41e817d.8949e27a` to `config.json`, and
       rename `cce28882.e02bd57e.h5` to `tf_model.h5`. Notice that we omitted the `.json` when renaming, as we want to rename the actual model files, not these `.json` files containing URLs.
       Once the files are renamed, copy them to a folder of your choice (e.g., `my_model_files`).
       (With knowledge of the URLs, you can also download the three model files from the listed URLs to your `my_model_files` folder and rename them appropriately, if you prefer.)
2. Copy the folder you created in the previous step (e.g., `my_model_files`) to the machine with no internet connectivity and point *ktrain* to the folder:
   ```python
   import ktrain
   from ktrain import text
   t = text.Transformer('/tmp/my_model_files', maxlen=500, class_names=label_list)
   trn = t.preprocess_train(x_train, y_train)
   model = t.get_classifier()
   learner = ktrain.get_learner(model, train_data=trn, batch_size=8)
   learner.fit_onecycle(5e-5, 1)
   ```

Note that the above steps are typically only necessary if training a model on the machine with no internet connectivity.  
The [bug](https://github.com/huggingface/transformers/issues/5016) does not affect loading predictors on machines with no internet.
That is, if all you're doing is making the predictions on the machine with no internet connectivity, doing `p = ktrain.load_predictor('/tmp/path_to_predictor')` is sufficient
provided the cache folder (i.e. `<home_directory>/.cache/torch/transformers`), contains the required model files. The vocab file is typically the only thing that
needs to be present in the cache for these scenarios.


Note also that the local path you supply to `Transformer` is stored in `t.model_name`, where `t` is a `Preprocessor` instance.  If creating a `Predictor` and transferring it to another machine, you may need to update this path:
```python
predictor.preproc.model_name = 'path/to/predictor/on/new/machine'
```



#### Example 2: Open-Domain QA (with no internet)


Here is a second example of how to run `SimpleQA` for [open-domain question-answering](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/question_answering_with_bert.ipynb) without internet access:

1. On a machine with public internet access, go to the Hugging Face model repository: [https://huggingface.co/models](https://huggingface.co/models)
2. Select the model you want and click "List all files in model".  For `SimpleQA`, you will need `bert-large-uncased-whole-word-masking-finetuned-squad` and `bert-base-uncased`
3. Download the `tf_model.h5`, `config.json`, and `vocab.txt` files into a folder.  It is important that these downloaded files are renamed specifically to the three aforementioned file names.
4. Copy these folders to the machine without public internet access
5. When invoking `SimpleQA`, provide these folders containing the downloaded files as arguments to the `bert_squad_model` and `bert_emb_model` parameters:
```python
qa = text.SimpleQA(INDEXDIR,
                    bert_squad_model='/path/to/bert/squad/model/folder',
                    bert_emb_model='/path/to/bert-base-uncased/folder')
```

You can use simlar steps for other models that use the `transformers` library like `bilstm-bert` for NER or offline language translation.



[[Back to Top](#frequently-asked-questions-about-ktrain)]

### How do I train using multiple GPUs?

Since *ktrain* is just a simple wrapper around TensorFlow,  you can use multiple GPUs in the same way you would for a normal `tf.Keras` model.
Here is a fully-complete, self-contained example for using 2 GPUs with a `Transformer` model:

```python
# use two GPUs to train
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0,1";  

# load text data
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
train_b = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
test_b = fetch_20newsgroups(subset='test',categories=categories, shuffle=True)
(x_train, y_train) = (train_b.data, train_b.target)
(x_test, y_test) = (test_b.data, test_b.target)

# build, train, and validate model
import tensorflow as tf
mirrored_strategy = tf.distribute.MirroredStrategy()
import ktrain
from ktrain import text
BATCH_SIZE = 6 * 2 # desired BS times 2
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=train_b.target_names)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
with mirrored_strategy.scope():
    model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, batch_size=BATCH_SIZE)
learner.fit_onecycle(5e-5, 2)
learner.save_model('/tmp/my_model')
learner.load_model('/tmp/my_model', preproc=t)
learner.validate(val_data=val, class_names=t.get_classes())
```

[[Back to Top](#frequently-asked-questions-about-ktrain)]



### How do I train a model using mixed precision?

See [this post](https://github.com/amaiya/ktrain/issues/126#issuecomment-616545655).


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I deploy a model using Flask?

First, implement the Flask server with something like this:

```python
# my_server.py
import flask
import ktrain
app = flask.Flask(__name__)
predictor = None
def load_predictor():
    global predictor
    predictor = ktrain.load_predictor('/tmp/my_saved_predictor')

@app.route('/predict', methods=['GET'])
def predict():
    data = {"success": False}
    if flask.request.method in ["GET"]:
        text = flask.request.args.get('text')
        if text is None: return flask.jsonify(data)
        prediction = predictor.predict(text)
        data['prediction'] = prediction
        data["success"] = True
    return flask.jsonify(data)

if __name__ == "__main__":
    load_predictor()
    port =8888 
    app.run(host='0.0.0.0', port=port)
    app.run()

```

Note that `/tmp_my_saved_predictor` is the path you supplied to `predictor.save`.  The `predictor.save` method
stores both the model and a `.preproc` object, so make sure both exist on the deployment server.

Next, start the server with: `python3 my_server.py`.

Finally, point your browser to the following to get a prediction:

```
http://0.0.0.0:8888/predict?text=text%20you%20want%20to%20classify
```

In this toy example, we are supplying the text data to classify in the URL as a GET request.

Note that the above example requires both **ktrain** and TensorFlow to be installed on the deployment machine.  If this footprint is too large,
you can [convert the model to ONNX](#how-do-i-make-quantized-predictions-with-transformers-models).  This allows you to deploy the model
and make predictions **without** having  **TensorFlow**, **ktrain**, and their many dependencies installed.  This is particurly well-suited to Heroku deployments, which restrict slug sizes to 500MB.



[[Back to Top](#frequently-asked-questions-about-ktrain)]

### How do I use custom metrics with *ktrain*?

The `Transformer.get_classifier`, `text.text_classifier`, and `vision.image_classifier` methods/functions all accept a `metrics` argument.

You can also use custom Keras callbacks:

```python
# define a custom callback for ROC-AUC
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
RocAuc = RocAucEvaluation(validation_data=(x_test, y_test), interval=1)

# train using our custom ROC-AUC callback
learner = ktrain.get_learner(model, train_data=train_data, val_data = val_data)
learner.autofit(0.005, 2, callbacks=[RocAuc])
```

[[Back to Top](#frequently-asked-questions-about-ktrain)]

### How do I get the predicted class "probabilities" of a model?

All `predict` methods in `Predictor` instances accept a `return_proba` argument.  Set it to true to obtain the class probabilities.


### How do I handle imbalanced datasets?

All `*fit*` methods (e.g., `learner.fit`, `learner.autofit`, `learner.fit_onecycle`) accept a `class_weight` parameter, which is passed
to the `model.fit` method in `tf.Keras`.  See [this StackOverflow post](https://stackoverflow.com/questions/44716150/how-can-i-assign-a-class-weight-in-keras-in-a-simple-way) for more details.

Alternatively, you can also try using [focal loss](https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/):

```python
import tensorflow as tf
from tensorflow.keras import activations
def focal_loss(gamma=2., alpha=4., from_logits=False):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax if from_logits is False.
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        if from_logits:
            y_pred = activations.softmax(y_pred)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed
```

As mentioned in [this issue](https://github.com/amaiya/ktrain/issues/228#issuecomment-672972996), you must use `from_logits=True`
if using `focal_loss` with a `transformers` model like DistilBert.


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I use custom loss functions or optimizers?

*ktrain* is just a lightweight wrapper around `tf.keras`, so this would be done in the exact same way as you would in Keras.
More specifically, you can simply recompile your model with the loss function or optimizer you want by invoking `model.compile`.

For example, here is how to use [focal loss](https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/) with a DistilBert model:

```python
import tensorflow as tf
from tensorflow.keras import activations
def focal_loss(gamma=2., alpha=4., from_logits=False):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax if from_logits is False.
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        if from_logits:
            y_pred = activations.softmax(y_pred)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

# load text data
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
train_b = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
test_b = fetch_20newsgroups(subset='test',categories=categories, shuffle=True)
(x_train, y_train) = (train_b.data, train_b.target)
(x_test, y_test) = (test_b.data, test_b.target)

# preprocess data and build model
import ktrain
from ktrain import text
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=train_b.target_names)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
model = t.get_classifier()

# recompile model with custom loss function
# using from_logits=True because output of transformer models are not run through softmax beforehand
model.compile(loss=focal_loss(alpha=1, from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

# train with focal loss
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
learner.fit_onecycle(5e-5, 1)
```


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I retrieve or visualize training history?

As with normal `tf.Keras` models, all `*fit*` methods in *ktrain* return the training history data.
```python
history = learner.autofit(...)
```

To visualize the training and validation loss by epochs:
```python
learner.plot('loss')
```

To visualize the learning rate schedule, you can do this:
```python
learner.plot('lr')
```

[[Back to Top](#frequently-asked-questions-about-ktrain)]

### I have a model that accepts multiple inputs (e.g., both text and other numerical or categorical variables).  How do I train it with *ktrain*?

See [this tutorial](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A4-customdata-text_regression_with_extra_regressors.ipynb).


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### Can I use `tf.data.Dataset` instances with *ktrain*?

Yes, but you'll need to wrap your dataset in a `ktrain.Dataset` instance, so that *ktrain* can more easily inspect your data.  
For instance, you can directly wrap a `tf.data.Dataset` instance as a `ktrain.TFDataset`, as shown in [this example](https://github.com/amaiya/ktrain/blob/master/examples/vision/mnist-tf_workflow.ipynb).

See [this tutorial](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A4-customdata-text_regression_with_extra_regressors.ipynb) for more information.


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### Why am I seeing a "list index out of range" error when calling predict?

The set of integer labels in your training set need to be complete and consecutive (e.g., `[0,1]` or `[0,1,2,3,4]`, but not `[0, 3]`). See [this post](https://github.com/amaiya/ktrain/issues/116#issuecomment-614864565).


[[Back to Top](#frequently-asked-questions-about-ktrain)]



### Why am I seeing an ERROR when installing *ktrain* on Google Colab?

These errors (e.g., `tensorboard 2.1.1 requires setuptools>=41.0.0, but you'll have setuptools 39.0.1 which is incompatible.`) are related to TensorFlow and can be usually safely ignored and shouldn't affect operation of *ktrain*.  The errors should go away if you perform the indicated upgrades (e.g., `pip install -U setuptools`).

[[Back to Top](#frequently-asked-questions-about-ktrain)]



### Running `predictor.explain` for text classification is slow.  How can I speed it up?

The `TextPredictor.explain` method accepts a parameter called `n_samples`, which governs the number of synthetic samples created and used to generate the explanation.  At the default value of 2500, `explain` returns results on Google Colab in ~25 seconds.
If you pass `n_samples=500` to `explain`, results are returned in ~5 seconds on Google Colab.  In theory, higher sample sizes  yield better explanations. In practice,
smaller sample sizes (e.g., 500, 1000) may be sufficient for your use case.


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### Running `preprocess_train` for Transformer models is slow.  How can I speed it up?

Preprocessing data for `transformers` text classification models using the `Transformer` API typically looks something like this:
```python
from ktrain import text
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=label_names)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
```
The `preprocess_train` and `preprocess_test` methods are not currently parallelized to use multiple CPU cores.  Some users have used [dask](https://github.com/dask/dask) to parallelize the preprocessing using something like this:

```python
import dask
def preproc(x, labels = labels):
    MODEL_NAME = 'distilbert-base-uncased'
    t = text.Transformer(MODEL_NAME, maxlen=80, class_names = labels, multilabel=True)
    res = t.preprocess_train(x['text_a'].values.tolist(),x['label'].values.tolist(), verbose=0)
    return(res)

results = []
partitions = train.to_delayed()
for part in partitions:
    results.append(dask.delayed(preproc)(part))

results = client.compute(results)

trn = results[0].result()
x = [r.result().x for r in results]
y = [r.result().y for r in results]
numlabels = np.max([yy.shape[1] for yy in y])
y = [np.pad(yy,[0,numlabels - yy.shape[1]], 'constant', constant_values = 0) for yy in y]
trn.x  = np.concatenate(x, axis = 0)
trn.y  = np.concatenate(y, axis = 0)
```

Note, however, that the power of transfer learning is being able to use smaller training sets to fine-tune your model. So, perhaps make sure you really need an extremely large training set before you try parallelizing the preprocessing.

[[Back to Top](#frequently-asked-questions-about-ktrain)]


### Why does `texts_from_csv` throw an error on Google Cloud Storage?

The error is probably happening because *ktrain* tries to auto-detect the character encoding using `open(train_filepath, 'rb')` which may be problematic with Google Cloud Storage. 
One solution is to explicitly provide the `encoding` to `texts_from_csv` as an argument so this step is skipped (default is *None*, which activates auto-detect).

Alternatively, you can read the data in yourself as a *pandas* DataFrame using one of [these methods](https://stackoverflow.com/a/50201179/13550699). For instance, *pandas* evidently supports GCS, so you can simply do this: `df = pd.read_csv('gs://bucket/your_path.csv')
`

Then, using *ktrain*, you can use `ktrain.text.texts_from_df` (or `ktrain.text.texts_from_array`) to load and preprocess your data.


[[Back to Top](#frequently-asked-questions-about-ktrain)]

### Why am I getting a 404 client error?

You can safely ignore the error, if it arises from downloading Hugging Face **transformers** models.  The 404 error simply means that **ktrain** was not able to find a Tensorflow version of this particular model. In this case, the PyTorch version of the  model checkpoint will be downloaded  and then be loaded by **ktrain** as a Tensorflow model for training/fine-tuning. If you type `model.summary()`, it should show that the model was loaded successfully.

[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I use ktrain with documents in PDF, DOC, or PPT formats?

If you have documents in formats like `.pdf`, `.docx`, or `.pptx` formats and want to use them in a training set or with various **ktrain** features 
like zero-shot-learning or text summarization, they will need to be converted to plain text format first (i.e., `.txt` files).  You can use the
`ktrain.text.textutils.extract_copy` function to automatically do this. Alternatively, you can use other tools like [Apache Tika](https://tika.apache.org/) to do the conversion.

With respect to Question-Answering, the `SimpleQA.index_from_folder` method includes a `use_text_extraction` argument.  When set to `True`, question-answering can be performed on document sets 
comprised of many different file types. More information on this is included in the [question-answering example notebook](https://github.com/amaiya/ktrain/blob/master/examples/text/question_answering_with_bert.ipynb).

[[Back to Top](#frequently-asked-questions-about-ktrain)]


### Can I use ktrain without a GPU?

Each task in **ktrain** offers different model choices.  Large models (e.g., fine-tuning BERT for text classification) definitely do require a GPU unless you have the patience for an unbearably slow training process.  However, smaller models (which can often yield very good accuracy scores), can be trained on a normal laptop CPU.   Examples of CPU-friendly models include the `nbsvm` model for text classification,  the `pretrained_mobilenet` model for image classification, [topic modeling](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-05-learning_from_unlabeled_text_data.ipynb), and models in the [ShallowNLP module](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/shallownlp-examples.ipynb).


A number of models in **ktrain** can be used out-of-the-box on a CPU-based laptop with no training required such as [question-answering](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/question_answering_with_bert.ipynb), [language translation](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/language_translation_example.ipynb), and [zero-shot topic classification](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/zero_shot_learning_with_nli.ipynb).



[[Back to Top](#frequently-asked-questions-about-ktrain)]



### How do I make quantized predictions with `transformers` models?

Quantization can improve the efficiency of neural network computations by reducing the size of the weights.  For instance, when making predictions, representing weights with 8-bit integers instead of 32-bit floats can speed up inferences.

TensorFlow has built-in support for quantization.  Unfortunately, as of this writing, it [only works for sequential and functional](https://github.com/tensorflow/tensorflow/issues/40699) `tf.keras` models, which means it cannot be used with Hugging Face `transformers` models.

As a workaround, you can convert your saved TensorFlow model to PyTorch, quantize, and make predictions directly in PyTorch. 

This code example assumes you've trained a DistilBERT model with **ktrain** ,saved a `Predictor` in a folder called `'/tmp/mypredictor'`, and need to make quantized predictions on CPU:
```python
# Quantization Using PyTorch

# load the predictor, model, and tokenizer
from transformers import *
import ktrain
predictor = ktrain.load_predictor('/tmp/mypredictor')
model_pt = AutoModelForSequenceClassification.from_pretrained('/tmp/mypredictor', from_tf=True)
tokenizer = predictor.preproc.get_tokenizer() # or use AutoTokenizer.from_pretrained(predictor.preproc.model_name)
maxlen = predictor.preproc.maxlen
device = 'cpu'
class_names = predictor.preproc.get_classes()

# quantize model (INT8 quantization)
import torch
model_pt_quantized = torch.quantization.quantize_dynamic(
    model_pt.to(device), {torch.nn.Linear}, dtype=torch.qint8)

# make quantized predictions (x_test is a list of strings representing documents)
preds = []
for doc in x_test:
    model_inputs = tokenizer(doc, return_tensors="pt", max_length=maxlen, truncation=True)
    model_inputs_on_device = { arg_name: tensor.to(device) 
                              for arg_name, tensor in model_inputs.items()}
    pred = model_pt_quantized(**model_inputs_on_device)
    preds.append(class_names[ np.argmax( np.squeeze( pred[0].cpu().detach().numpy() ) ) ]) 

```

Note that the above example employs smaller inputs by eliminating padding in addition to using a quantized model.  As discussed in [this blog post](https://blog.roblox.com/2020/05/scaled-bert-serve-1-billion-daily-requests-cpus/), both of these steps can speed up predictions in CPU deployment scenarios.

Alternatively, you might also consider quantizing your `transformers` model with the [convert_graph_to_onnx.py](https://github.com/huggingface/transformers/blob/master/src/transformers/convert_graph_to_onnx.py) script included with the `transformers` library, which can also be used as a module, as shown below.

```python
# Converting to ONNX (from PyTorch-converted model)


# set maxlen, class_names, and tokenizer (use settings employed when training the model - see above)
model_name = 'distilbert-base-uncased'
maxlen = 500
class_names = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


# imports
import numpy as np
from transformers.convert_graph_to_onnx import convert, optimize, quantize
from transformers import AutoModelForSequenceClassification
from pathlib import Path

# paths
predictor_path = '/tmp/mypredictor'
pt_path = predictor_path+'_pt'
pt_onnx_path = pt_path +'_onnx/model.onnx'

# convert to ONNX
AutoModelForSequenceClassification.from_pretrained(predictor_path, 
                                                   from_tf=True).save_pretrained(pt_path)
convert(framework='pt', model=pt_path,output=Path(pt_onnx_path), opset=11, 
        tokenizer=model_name, pipeline_name='sentiment-analysis')
pt_onnx_quantized_path = quantize(optimize(Path(pt_onnx_path)))

# create ONNX session
def create_onnx_session(onnx_model_path, provider='CPUExecutionProvider'):
    """
    Creates ONNX inference session from provided onnx_model_path
    """

    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 0
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend 
    session = InferenceSession(onnx_model_path, options, providers=[provider])
    session.disable_fallback()
    return session
sess = create_onnx_session(pt_onnx_quantized_path.as_posix())

#  tokenize document and make prediction
tokens = tokenizer.encode_plus('My computer monitor is blurry.', max_length=maxlen, truncation=True)
tokens = {name: np.atleast_2d(value) for name, value in tokens.items()}
print()
print()
print("predicted class: %s" % (class_names[np.argmax(sess.run(None, tokens)[0])]))

# output:
# predicted class: comp.graphics
```

The example above assumes the model saved at `predictor_path` was trained on a subset of the 20 Newsgroup corpus as was done in [this tutorial](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A3-hugging_face_transformers.ipynb).

You can also use **ktrain** to create ONNX models directly from TensorFlow with (which can be used for non-transformers TensorFlow models): 
```python
predictor.export_model_to_onnx(onnx_model_path)
```

However, note that conversions to ONNX from TensorFlow models appear to [require a hard-coded input size](https://github.com/huggingface/transformers/issues/8227) (i.e., padding is used), whereas conversions to ONNX from PyTorch models do not appear to have this requirement.


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I train a transformers model from a saved checkpoint folder?

In the **ktrain** `Transformer` API, you can train/fine-tune a text classification model from a local path:
```python
t = text.Transformer(MODEL_LOCAL_PATH, maxlen=50, class_names=class_names)
```

This is useful, for example, if you first [fine-tune a language model](https://github.com/huggingface/transformers/tree/master/examples/language-modeling) using Hugging-Face **Trainer** **prior** to fine-tuning your text classifier.

However, when supplying a local path to `Transformer`, **ktrain** will also look for the tokenizer files in that directory. So, you just need to ensure tokenizer files like the `vocab.txt` (which are quite small), exist in the local folder (and also exist in the folder created by `predictor.save_predictor`.  Such files can be downloaded from the Hugging Face model hub.  See [this post](https://github.com/amaiya/ktrain/issues/295#issuecomment-744509996) and [this FAQ entry](https://github.com/amaiya/ktrain/blob/master/FAQ.md#how-do-i-use-ktrain-without-an-internet-connection) for more details.

Note that the local path you supply to `Transformer` is stored in `t.model_name`, where `t` is a `Preprocessor` instance.  If creating a `Predictor` and transferring it to another machine, you may need to update this path:
```python
predictor.preproc.model_name = 'path/to/predictor/on/new/machine'
```



[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I pretrain a language model for use with ktrain?

It is very easy to pretrain a `transformer` language model (either fine-tuning the language model or training from scratch) using [this Hugging Face script](https://github.com/huggingface/transformers/tree/master/examples/language-modeling).  This can sometimes boost performance especially if your dataset has highly specialized terminology.

These Hugging Face scripts will save the fine-tuned pretrained language model to a folder.  One can then simply point **ktrain** to this folder to fine-tune a text-classifier using this fine-tuned/pretrained language model using either of the following two approaches:


#### Approach 1 
You need to copy tokenizer files (which are very small) to the path of the saved language model. These files can be obtained from the Hugging Face model hub. This is also required when loading models without an internet connection, as described in [this FAQ entry](https://github.com/amaiya/ktrain/blob/master/FAQ.md#how-do-i-use-ktrain-without-an-internet-connection).

Note that, when you save the `Predictor` to a folder,  you'll again need to make sure that folder  has the tokenizer files.  Otherwise, `predictor.predict` will yield the same errors.


#### Approach 2 
Alternatively, you could try loading the tokenizer yourself with **transformers** and  manually setting the `t.tok=tokenizer` prior to calling `preprocess_train`:

```python
t = text.Transformer(MODEL_LOCAL_PATH, maxlen=50, class_names=class_names)
from transformers import *
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
t.tok = tokenizer
t.preprocess_train(...
```
When loading a predictor, you'll also need to reset tokenizer manually:
```python
p = ktrain.load_predictor('/tmp/mypred')
p.preproc.tok = tokenizer
p.predict('Some text to predict')
```

Note that the local path you supply to `Transformer` is stored in `t.model_name`, where `t` is a `Preprocessor` instance.  If creating a `Predictor` and transferring it to another machine, you may need to manually update this path:
```python
predictor.preproc.model_name = 'path/to/predictor/on/new/machine'
```


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I get reproducible results?

In regard to train-test splits, the data-loading functions (e.g., `texts_from_folder`, `images_from_csv`) have a `random_state` parameter that will ensure the same dataset split across runs.

In regards to training, please see [this post](https://github.com/amaiya/ktrain/issues/334#issuecomment-788893119), which includes some suggestions for reproducible results in `tf.keras` and TensorFlow 2.

For instance, invoking the function below before each training run can help generate more consistent results across runs.

```python
import tensorflow as tf
import numpy as np
import os
import random
def reset_random_seeds(seed=2):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
```


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I increase batch size for predictions?

Increasing the batch size used for inference and predictions can potentially speed up predictions on lists of examples.

The `get_predictor` and `load_predictor` functions both accept a `batch_size` argument that will be used when making predictions on lists of examples. The default is 32.  The `batch_size` for `Predictor` instances can also be set manually:
```python
predictor = ktrain.load_predictor('/tmp/my_predictor')
predictor.batch_size = 128
predictor.predict(list_of_examples)
```

The `get_learner` function accepts an `eval_batch_size` argument that will be used by the `Learner` instance when evaluating a validation dataset (e.g., `learner.predict`).


[[Back to Top](#frequently-asked-questions-about-ktrain)]



### How do I do cross validation with transformers?

Here is a quick self-contained example:

```python
from ktrain import text
import ktrain
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups

# load text data
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
train_b = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
test_b = fetch_20newsgroups(subset='test',categories=categories, shuffle=True)
(x_train, y_train) = (train_b.data, train_b.target)
(x_test, y_test) = (test_b.data, test_b.target)
df = pd.DataFrame({'text':x_train, 'target': [train_b.target_names[y] for y in y_train]})

# CV with transformers
N_FOLDS = 2
EPOCHS = 3
LR = 5e-5
def transformer_cv(MODEL_NAME):
    preproc  = text.Transformer(MODEL_NAME, maxlen=500)
    predictions,accs=[],[]
    data = df[['text', 'target']]
    for train_index, val_index in KFold(N_FOLDS).split(data):
        preproc  = text.Transformer(MODEL_NAME, maxlen=500)
        train,val=data.iloc[train_index],data.iloc[val_index]
        x_train=train.text.values
        x_val=val.text.values

        y_train=train.target.values
        y_val=val.target.values

        trn = preproc.preprocess_train(x_train, y_train)
        model = preproc.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, batch_size=16)
        learner.fit_onecycle(LR, EPOCHS)
        predictor = ktrain.get_predictor(learner.model, preproc)
        pred=predictor.predict(x_val)
        acc=accuracy_score(y_val,pred)
        print('acc',acc)
        accs.append(acc)
    return accs
print( transformer_cv('distilbert-base-uncased') )
```


[[Back to Top](#frequently-asked-questions-about-ktrain)]




### What kinds of applications have been built with *ktrain*?

Examples include:

- **medical informatics:**  analyzing doctors' written analyses of patients and medical imagery
- **finance:**  financial crime analytics, mining stock-related news stories
- **insurance:** detecting fraud in insurance claims
- **customer relationship management (CRM):** making sense of feedback from customers and/or patients
- **political science:** study on targeted political messaging
- **news media:** prioritizing political claims for fact-checking
- **social science:** making sense of text-based responses in surveys and emotion-classification from text data
- **linguistics:** detecting sarcasm in the news
- **education:** analysis of attitudes towards educational institutions in social media
- **local government:**: auto-categorizing citizen complaints to local governments
- **federal government:** extracting insights from documents about government programs and policies

[[Back to Top](#frequently-asked-questions-about-ktrain)]



