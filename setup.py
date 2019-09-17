from distutils.core import setup
import setuptools

with open('README.md') as readme_file: readme = readme_file.read()
exec(open('ktrain/version.py').read())

setup(
  name = 'ktrain',
  packages = setuptools.find_packages(),
  version = __version__,
  license='MIT',
  description = 'ktrain is a lightweight wrapper for Keras to help train neural networks',
  long_description = readme,
  long_description_content_type = 'text/markdown',
  author = 'Arun S. Maiya',
  author_email = 'arun@maiya.net',
  url = 'https://github.com/amaiya/ktrain',
  keywords = ['keras', 'deep learning', 'machine learning'],
  install_requires=[
          'keras == 2.2.4',
          'scikit-learn >= 0.20.0',
          'matplotlib >= 3.0.0',
          'pandas >= 0.24.2',
          'fastprogress >= 0.1.21',
          'keras_bert',
          'requests',
          'eli5 >= 0.10.0',
          'seqeval',
          'joblib'
      ],
  classifiers=[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

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
