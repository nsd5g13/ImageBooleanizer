# Image Dataset Booleanizer for Logic-Based Machine Learning (e.g., Tsetlin Machine)

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

Install pyTsetlinMachine (for Tsetlin Machine training, original source code available at [https://github.com/cair/pyTsetlinMachine](https://github.com/cair/pyTsetlinMachine)):

```sh
   cd pyTsetlinMachine
   python setup.py install
   ```

## Usage

### Booleanization

A logic-based machine learning algorithm, such as the Tsetlin Machine (TM), typically requires input features to be represented as Boolean values. We provide source code for Booleanizing multiple open-source image classification datasets.

* [Digits](https://scikit-learn.org/1.5/auto_examples/datasets/plot_digits_last_image.html)
* [MNIST](https://keras.io/api/datasets/mnist/)
* [Fashine-MNIST (FMNIST)](https://keras.io/api/datasets/fashion_mnist/)
* [Kuzushiji-MNIST (KMNIST)](https://github.com/rois-codh/kmnist)
* [CIFAR10](https://keras.io/api/datasets/cifar10/)
* CIFAR2: A binary classification variant of CIFAR-10, where the original ten classes are grouped into Animal and Non-Animal categories.
* [Keyword Spotting (KWS)](https://www.kaggle.com/datasets/antfilatov/mini-speech-commands)

Before Booleanizing, download KMNIST and KWS datasets and put the dataset directories at "raw\_datasets/".

```sh
   python booleanization.py [dataset_name]
   ```

Options for [dataset_name] are digits, mnist, fmnist, kmnist, cifar10, cifar2, kws and all, where "all" suggests producing Booleanized datasets for all above.

### Training TM models for the Booleanized datasets

```sh
   usage: TM_Training.py clauses T s epochs budget dataset_name

 positional arguments:
     clauses         Provide the number of clauses per class
     T               Provide the value of "Threshold"
     s               Provide the value of "Strength" for literal include
     epochs          Proivde the number of training epochs
     budget          Provide the constrain for the maximal number of literals included in each clause
     dataset_name    Provide the name of the dataset. Options include digits, mnist, fmnist, kmnist, cifar10, cifar2, kws
   ```

Example:

```sh
   python TM_Training.py 100 10 3 100 1568 MNIST
   ```

<!-- PAPER -->

## Paper

* Y. Zeng, S. Duan, R. Shafik, and A. Yakovlev, "Inference Latency-Aware Tsetlin Machine Training," in 5th International Symposium on the Tsetlin Machines (ISTM), 2026.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Preprocessing -->

## How each dataset is proprocessed

### MNIST, FMNIST and KMNIST
* A specific threshold (75) is applied, converting each of the 784 raw features into a 1-bit value across all datapoints.

### Digits
* A specific threshold (7.5) is applied, converting each of the 64 raw features into a 1-bit value across all datapoints.

### CIFAR
* Feature extraction: Histogram of Oriented Gradients (HOG) is applied to extract high-level features from the raw. HOG parameters are as follows:
- Window size: 32x32 pixels
- Block size: 24x24 pixels
- Block stride: 8x8 pixels
- Cell size: 8x8 pixels
- Number of bins: 9
324 features, 18 for each column/row, are extracted through HOG.
 
* Quantile binning is applied, converting each of the 324 extracted features into a 1-bit value across all datapoints.

### KWS

