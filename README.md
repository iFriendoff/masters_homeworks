# Machine Learning and Numerical Methods Homeworks

A collection of coursework notebooks and small Python implementations covering
machine learning, deep learning, natural language and audio processing, image
processing, and numerical methods.

## Homework notebooks

| Notebook | Description |
| --- | --- |
| [`DLHouseprice.ipynb`](./DLHouseprice.ipynb) | Builds a TensorFlow dense neural network to predict house prices from tabular features. |
| [`FineTuningResNet.ipynb`](./FineTuningResNet.ipynb) | Trains image classifiers for flower categories, including transfer learning and fine-tuning with EfficientNet. |
| [`KNNNNNN.ipynb`](./KNNNNNN.ipynb) | Applies a custom vectorized k-nearest-neighbors classifier to a tabular classification dataset. |
| [`minMSE.ipynb`](./minMSE.ipynb) | Explores mean-squared-error minimization and fits a vectorized linear model to tabular data. |
| [`MNIST.ipynb`](./MNIST.ipynb) | Creates a TensorFlow feed-forward neural network for handwritten-digit classification. |
| [`MY_ML_MODELS.ipynb`](./MY_ML_MODELS.ipynb) | Demonstrates custom linear regression, logistic regression, perceptron, naive Bayes, and KNN models on example datasets. |
| [`Nayiv_Bayes.ipynb`](./Nayiv_Bayes.ipynb) | Implements naive Bayes classification for predicting Titanic passenger survival. |
| [`NumericalLecture5Saidel.ipynb`](./NumericalLecture5Saidel.ipynb) | Demonstrates the iterative Gauss-Seidel method for solving a system of linear equations. |
| [`NumericalMethods3lecture.ipynb`](./NumericalMethods3lecture.ipynb) | Implements numerical root-finding approaches, including interval-based and fixed-point iteration methods. |
| [`numerical_Gauss.ipynb`](./numerical_Gauss.ipynb) | Solves a system of linear equations using Gaussian elimination. |
| [`poool.ipynb`](./poool.ipynb) | Explores breast-cancer diagnosis classification and distance-based neighbor calculations. |
| [`preprocessing.ipynb`](./preprocessing.ipynb) | Uses OpenCV for image filtering, noise comparison, PSNR measurement, thresholding, and contour detection. |
| [`seminar6.ipynb`](./seminar6.ipynb) | Demonstrates microphone-based speech recognition with Python's speech-recognition library. |
| [`vectorizedmodels.ipynb`](./vectorizedmodels.ipynb) | Works through vectorized machine-learning model calculations with NumPy. |
| [`NLP/NLP1.ipynb`](./NLP/NLP1.ipynb) | Examines sampled signals and their frequency components using the Fourier transform. |
| [`NLP/NLP2.ipynb`](./NLP/NLP2.ipynb) | Analyzes audio signals, waveforms, and spectrograms using Python audio-processing libraries. |
| [`NLP/NLP3.ipynb`](./NLP/NLP3.ipynb) | Builds a PyTorch convolutional recurrent neural network for recognizing sequences of handwritten digits. |
| [`NLP/NLP5.ipynb`](./NLP/NLP5.ipynb) | Implements text normalization, tokenization, alignment, and word-error-rate scoring. |

## Python implementations

- [`my_ml_vectorized_models.py`](./my_ml_vectorized_models.py) contains NumPy
  implementations of linear regression, logistic regression, KNN, naive Bayes,
  and a perceptron.
- [`vectorized_models.py`](./vectorized_models.py) is an empty placeholder in
  the current project.
- [`numerical/chord.py`](./numerical/chord.py) and
  [`numerical/dihotomy.py`](./numerical/dihotomy.py) demonstrate the chord and
  bisection methods for finding roots.

## Data and dependencies

Datasets, images, audio files, and other data assets are intentionally not
included. Some notebooks expect local files under `data/`, `NLP/data/`, or
`flower_photos/`; provide the corresponding files and adjust paths as needed
before running them. Install the libraries imported by a notebook in your
Python environment before execution.
