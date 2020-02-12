from distutils.core import setup
import setuptools

with open('README.md') as readme_file: 
    readme_file.readline()
    readme = readme_file.read()
exec(open('ktrain/version.py').read())


setup(
  name = 'ktrain',
  packages = setuptools.find_packages(),
  version = __version__,
  license='MIT',
  description = 'ktrain is a lightweight wrapper for TensorFlow Keras to help train neural networks',
  long_description = readme,
  long_description_content_type = 'text/markdown',
  author = 'Arun S. Maiya',
  author_email = 'arun@maiya.net',
  url = 'https://github.com/amaiya/ktrain',
  keywords = ['tensorflow', 'keras', 'deep learning', 'machine learning'],
  install_requires=[
          'scikit-learn == 0.21.3',
          'matplotlib >= 3.0.0',
          'pandas < 1.0',
          'fastprogress >= 0.1.21',
          'keras_bert',
          'requests',
          'joblib',
          'langdetect',
          'jieba',
          'cchardet',
          'networkx==2.3',
          'bokeh',
          'seqeval',
          'packaging',
          'tensorflow_datasets',
          'transformers',
          'ipython'
          #'stellargraph>=0.8.2',
          #'eli5 >= 0.10.0',
          #'pillow'
      ],
  classifiers=[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
