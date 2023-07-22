import sys

if sys.version_info.major != 3:
    raise Exception("ktrain requires Python 3")

from distutils.core import setup

import setuptools

with open("README.md", encoding="utf-8") as readme_file:
    readme_file.readline()
    readme = readme_file.read()
exec(open("ktrain/version.py").read())

all_extras = [
    # "torch",  # for qa, summarization, translation, zsl, speech
    "ipython",  # for tests
    "textract",  # for TextExtractor non-default method
    "datasets",  # for text.qa.AnswerExtractor.finetune
    "textblob",  # for text.kw.KeywordExtractor and textutils.extract_noun_phrases
    "sumy",  # text.summarization.core.LexRankSummarizer
    "causalnlp",  # for tabular.causalinference
    "librosa",  # for text.speech
    "shap",  # for tabular.TabularPredictor.explain
    "eli5 @ git+https://github@github.com/amaiya/eli5-tf@master#egg=eli5",  # for explain in text/vision
    "stellargraph @ git+https://github@github.com/amaiya/stellargraph@no_tf_dep_082#egg=stellargraph",  # for graph module
]
# not included and checked/requested within-code:
# 1. bokeh: in TopicModel.visualize_docuemnts
# 2. allennlp: for NETR Elmo embeddings since TF2 TF_HUB does not work

setup(
    name="ktrain",
    packages=setuptools.find_packages(),
    package_data={"ktrain": ["text/shallownlp/ner_models/*", "text/stopwords-zh.txt"]},
    version=__version__,
    license="Apache License 2.0",
    description="ktrain is a wrapper for TensorFlow Keras that makes deep learning and AI more accessible and easier to apply",
    # description = 'ktrain is a lightweight wrapper for TensorFlow Keras to help train neural networks',
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Arun S. Maiya",
    author_email="arun@maiya.net",
    url="https://github.com/amaiya/ktrain",
    keywords=["tensorflow", "keras", "deep learning", "machine learning"],
    install_requires=[
        "scikit-learn",
        "matplotlib >= 3.0.0",
        "pandas >= 1.0.1",
        "fastprogress >= 0.1.21",
        "requests",
        "joblib",
        "packaging",
        "langdetect",
        "jieba",
        "cchardet",  # previously pinned to 2.1.5 (due to this issue: https://github.com/PyYoshi/cChardet/issues/61) but no longer needed
        "chardet",  # required by stellargraph and no longer installed by other dependencies as of July 2021
        "syntok>1.3.3",  # previously pinned to 1.3.3 due to bug in 1.4.1, which caused problems in QA paragraph tokenization (now appears fixed)
        "tika",  # for TextExtractor
        # NOTE: these modules can be optionally omitted from deployment if not being used to yield lighter-weight footprint
        "transformers>=4.17.0",  # previously pinned to transformers==4.17.0 due to undocumented transformers issue in >=4.18.0 -> **kwargs removed from call
        "sentencepiece",  #  Added due to breaking change in transformers>=4.0
        "keras_bert>=0.86.0",  # imported in imports with warning and used in 'ktrain.text' ; support for TF 2.3
        "whoosh",  # imported by text.qa module
        "paper-qa==2.1.1",  # text.qa.generative_qa
    ],
    extras_require={
        # NOTE: If missing, these libraries below are installed manually on-the-fly when required by an invoked method with appropriate warnings
        # for testing: pip3 install git+https://github@github.com/amaiya/ktrain@develop#egg=ktrain[tests]
        "tests": all_extras,
        "all": all_extras,
    },
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
