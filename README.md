# CNN-Model-for-Fashion-MNIST-dataset

## Introduction

This code implements a TensorFlow 2.x model to classify fashion items from the Fashion-MNIST dataset. The model uses a Convolutional Neural Network (Conv2D) and a Dense layer for the prediction. The dataset is first preprocessed by normalizing and reshaping the data and then one-hot encoding the labels. The model is compiled with Adam optimizer, categorical crossentropy loss, and accuracy as the metric. The model is trained for 128 epochs with a batch size of 128.

## Libraries Used

    Numpy
    Matplotlib
    Pandas
    TensorFlow

## Dataset

The data is stored in two csv files: fashion-mnist_train.csv and fashion-mnist_test.csv, which are loaded using Pandas. The data is normalized and reshaped to 28x28x1 arrays and then split into labels and features.

## Model

The model is a sequential model with a single Conv2D layer and a Dense layer. The Conv2D layer has 32 filters and a kernel size of 3x3. The Dense layer has 10 units and uses a softmax activation function. The model is compiled with Adam optimizer and categorical crossentropy loss. The accuracy is used as the metric.

## Training

The model is trained for 128 epochs with a batch size of 128. The training process outputs the loss and accuracy for each epoch.
