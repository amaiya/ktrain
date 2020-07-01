# Frequently Asked Questions About *ktrain*

- [I am a newcomer and am having trouble figuring out how to even get started. Where do I begin?](#i-am-a-newcomer-and-am-having-trouble-figuring-out-how-to-even-get-started-where-do-i-begin)

- [How do I resume training from a saved checkpoint?](#how-do-i-resume-training-from-a-saved-checkpoint)

- [How do I obtain the word or sentence embeddings after fine-tuning a Transformer-based text classifier?](#how-do-i-obtain-the-word-or-sentence-embeddings-after-fine-tuning-a-transformer-based-text-classifier)

- [How do I use ktrain without an internet connection?](#how-do-i-use-ktrain-without-an-internet-connection)

- [How do I train using multiple GPUs?](#how-do-i-train-using-multiple-gpus)

- [How do I train a model using mixed precision?](#how-do-i-train-a-model-using-mixed-precision)

- [How do I deploy a model using Flask?](#how-do-i-deploy-a-model-using-flask)

- [How do I use custom metrics with ktrain?](#how-do-i-use-custom-metrics-with-ktrain)

- [How do I get the predicted class "probabilities" of a model?](#how-do-i-get-the-predicted-class-probabilities-of-a-model)

- [How do I retrieve or visualize training history?](#how-do-i-retrieve-or-visualize-training-history)

- [I have a model that accepts multiple inputs (e.g., both text and other numerical or categorical variables).  How do I train it with *ktrain*?](#i-have-a-model-that-accepts-multiple-inputs-eg-both-text-and-other-numerical-or-categorical-variables--how-do-i-train-it-with-ktrain)

- [Can I use `tf.data.Dataset` instances with *ktrain*?](#can-i-use-tfdatadataset-instances-with-ktrain)

- [Why am I seeing an ERROR when installing *ktrain* on Google Colab?](#why-am-i-seeing-an-error-when-installing-ktrain-on-google-colab)

- [Running `predictor.explain` for text classification is slow.  How can I speed it up?](#running-predictorexplain-for-text-classification-is-slow--how-can-i-speed-it-up)

- [Why does `texts_from_csv` throw an error on Google Cloud Storage?](#why-does-texts_from_csv-throw-an-error-on-google-cloud-storage)


- [What kinds of applications have been built with *ktrain*?](#what-kinds-of-applications-have-been-built-with-ktrain)



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


- For more information on `ktrain`, see [the tutorials](https://github.com/amaiya/ktrain#tutorials).

- For more information on Python, see [here](https://learnpythonthehardway.org/).

- For more information on neural networks, see [this page](https://victorzhou.com/blog/intro-to-neural-networks/).

- For more information on Google Colab, see [this video](https://www.youtube.com/watch?v=inN8seMm7UI).

- For more information on Jupyter notebooks, see [this video](https://www.youtube.com/watch?v=HW29067qVWk).

*ktrain* is inspired by some other libraries like `fastai` and `ludwig`. For a deeper dive into neural networks, the *fastai MOOC* and the
*TensorFlow and Deep Learning Without a PhD* series are recommended.


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I resume training from a saved checkpoint?

#### Method 1: Using Predictor API (works for any model)
```python
# save Predictor (i.e., model and Preprocessor instance) after partially training
ktrain.get_predictor(model, preproc).save('/tmp/my_predictor')

# reload Predictor and extract model
model = ktrain.load_predictor('/tmp/my_predictor').model

# re-instantiate Learner and continue training
learner = ktrain.get_learner(model, train_data=trn, val_data=val)
learner.fit_onecycle(2e-5, 1)
```
Note that `preproc` here is a *Preprocessor* instance.  If using a data-loading function like `texts_from_csv` or `images_from_folder`, it will be the third return value from the function. Or, if using the *Transformer* API, it will be the output of invoking `text.Transformer` (i.e., `preproc = text.Transformer('bert-base-uncased', ...)`).


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
learner = ktrain.get_learner(model, train_data=trn, val_data=val)
learner.fit_onecycle(2e-5, 1)
```

#### Method 3: Using `checkpoint_folder` argument to save model weights

The `checkpoint_folder` argument (e.g., `learner.autofit(1e-4, 4, checkpoint_folder='/tmp/saved_weights')`), saves the weights only of the model after each epoch. 
The weights of any epoch can be reloaded into the model using the `model.load_weights` method as you normally would in `tf.Keras`.  You just need to first re-create
the model first.  For instance, if training an NER model, it would work as follows:
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

[[Back to Top](#frequently-asked-questions-about-ktrain)]


### How do I use ktrain without an internet connection?

When using pretrained models or pretrained word embeddings in *ktrain*, files are automatically downloaded.  For instance,
pretrained models and tokenizers from the `transformers` library are downloaded to `<home_directory>/.cache/torch/transformers`
by default.  Other data like pretrained word vectors are downloaded to the `<home_directory>/ktrain_data` folder.

In some settings, it is necessary to either train models or make predictions in environments with no internet 
access (e.g., behind a firewall, air-gapped networks).  Typically, it is sufficient to copy the above folders
to the machine without internet access. For instance, if loading and using a `Predictor` instance as shown below,
then all that is typically needed is a vocabulary file that is typically retrieved from the cache:
```python
p = ktrain.load_predictor('/tmp/mypred')
p.predict(data)
```

In some cases (e.g., when training a model on a system with no internet access or using pretrained model for question-answering),
 due to a current bug in the `transformers` library, files from `<home_directory>/.cache/torch/transformers` may
not load when there is no internet access even when present.  To get around this, you can download the model files from [here]( https://huggingface.co/models) and point
*ktrain* to the folder.  There are typically three files you need, and it is important that the downloaded files are rennamed 
to `tf_model.h5`, `config.json`, and `vocab.txt`.

Here is an example of how to run `SimpleQA` for [open-domain question-answering](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/question_answering_with_bert.ipynb) without internet access:

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

You can use simlar steps for other models that use the `transformers` library like text classification using the `ktrain.text.Transformer` class, for example.



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


### Why am I seeing an ERROR when installing *ktrain* on Google Colab?

These errors (e.g., `has requirement gast>=0.3.2, but you'll have gast 0.2.2 which is incompatible`) are related to TensorFlow and can be usually safely ignored and shouldn't affect operation of *ktrain*.

[[Back to Top](#frequently-asked-questions-about-ktrain)]



### Running `predictor.explain` for text classification is slow.  How can I speed it up?

The `TextPredictor.explain` method accepts a parameter called `n_samples`, which governs the number of synthetic samples created and used to generate the explanation.  At the default value of 2500, `explain` returns results on Google Colab in ~25 seconds.
If you pass `n_samples=500` to `explain`, results are returned in ~5 seconds on Google Colab.  In theory, higher sample sizes  yield better explanations. In practice,
smaller sample sizes (e.g., 500, 1000) may be sufficient for your use case.


[[Back to Top](#frequently-asked-questions-about-ktrain)]


### Why does `texts_from_csv` throw an error on Google Cloud Storage?

The error is probably happening because *ktrain* tries to auto-detect the character encoding using `open(train_filepath, 'rb')` which may be problematic with Google Cloud Storage. 
One solution is to explicitly provide the `encoding` to `texts_from_csv` as an argument so this step is skipped (default is *None*, which activates auto-detect).

Alternatively, you can read the data in yourself as a *pandas* DataFrame using one of [these methods](https://stackoverflow.com/a/50201179/13550699). For instance, *pandas* evidently supports GCS, so you can simply do this: `df = pd.read_csv('gs://bucket/your_path.csv')
`

Then, using *ktrain*, you can use `ktrain.text.texts_from_df` (or `ktrain.text.texts_from_array`) to load and preprocess your data.


[[Back to Top](#frequently-asked-questions-about-ktrain)]

### What kinds of applications have been built with *ktrain*?

Examples include:

- **medical informatics:**  analyzing doctors' written analyses of patients and medical imagery
- **finance:**  financial crime analytics, mining stock-related news stories
- **insurance:** detecting fraud in insurance claims
- **customer relationship management (CRM):** making sense of feedback from customers and/or patients
- **social science:** making sense of text-based responses in surveys and emotion-classification from text data
- **linguistics:** detecting sarcasm in the news
- **education:** analysis of attitudes towards educational institutions in social media
- **local government:**: auto-categorizing citizen complaints to local governments
- **federal government:** extracting insights from documents about government programs and policies

[[Back to Top](#frequently-asked-questions-about-ktrain)]


