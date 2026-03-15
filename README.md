# MNIST Classification with PyTorch

A small convolutional neural network that classifies handwritten digits (0-9) on the MNIST dataset. The project was built to learn the PyTorch ecosystem: data loading, model definition, training loops, and inference.

## Setup

Create a virtual environment, install dependencies from `requirements.txt` with `pip install -r requirements.txt`, and run the notebook. The dataset is downloaded automatically on first run (into a `data/` directory, but feel free to change that to your liking).

## Architecture

The model is a CNN with two convolutional blocks followed by a classifier head.

**Block 1:** Two 3x3 convolutions (1 channel to 32, then 32 to 32), each with BatchNorm and ReLU. MaxPool 2x2 reduces the spatial size from 28x28 to 14x14.

**Block 2:** Two 3x3 convolutions (32 to 64, then 64 to 64), again with BatchNorm and ReLU. A second 2x2 MaxPool reduces 14x14 to 7x7.

**Classifier:** The 64x7x7 feature map is flattened, passed through a linear layer (3136 to 128), ReLU, dropout (0.5), then a linear layer to 10 classes (logits). No softmax in the forward pass; CrossEntropyLoss is used for training.

Input images are normalized with MNIST mean 0.1307 and std 0.3081 (single channel). Training uses Adam (lr 0.001) and 10 epochs; typical results are around 99.2%+ on both train and test (no validation set needed here).

## Usage

Train and evaluate in the notebook (data loaders, train/test functions, and training loop are in the notebook). For a single image, use the `infer(img_path, target)` function: it loads the image, applies the same preprocessing, runs the model, and prints the prediction and whether it matches the given target. Path and image validity are checked before inference.