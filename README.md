# Image classification using CNN
This project was developed for the Artificial Neural Network Project. It classifies images of cats and dogs using Convolutional Neural Network, with accuracy around 80%.


# Description of content

This directory contains:

* main.py
    + to describe the CNN model,
    + to train, evaluate and test the model,
    + to plot validation and training loss graphs, validation accuracy graph and confusion matrix for test set.

* grid_search_optimization.py 
    + to perform grid search,
    + to test and choose the best hyperparameters for the model,

* cnn_model.pth
    + this is the file with saved trained model.

* test.py 
    + use this file to test the trained saved model on exemplary dataset,
    + after finishing, it plots confusion matrix and prints accuracy,
    + follow undermentioned instructions to run the test file.

* dataset -- this is the directory with test set data, which are used in test file. It consists of 10 randomly chosen images of cats and 10 images of dogs. 

* full_dataset -- this is full dataset with directories to train, validate and test model. Each subdirectory contains another two directories with images of cats and dogs.

# Project setup instructions

## Step 1: Clone git repository

1. Clone git repository to a chosen directory.

## Step 2: Create and activate virtual environment

### On Linux:

1. Open Terminal and navigate to the project directory.
2. Create a virtual environment:
    ```bash
    python3 -m venv myenv
    ```
3. Activate the virtual environment:
    ```bash
    source myenv/bin/activate
    ```

## Step 3: Install required libraries

1. Ensure the virtual environment is activated.
2. Install the required libraries using `requirements.txt` (it may take a moment):
    ```bash
    pip install -r requirements.txt
    ```

## Step 4: Run the program

### test.py 

1. With the virtual environment activated, run the program test.py (it may take a moment):
    ```bash
    python3 test.py
    ```
As a result, it will print out the accuracy and draw a confusion matrix.
Please note that this file uses saved model "cnn_model.pth" and "dataset" with test set only. You can change those in lines 75-76:

    model_path = "cnn_model.pth"
    test_dir = "dataset/test_set"

### main.py

2. With the virtual environment activated, run the program main.py (it may take a moment):
    ```bash
    python3 main.py
    ```
As a result, it will plot validation and training loss graphs, validation accuracy graph and confusion matrix. Please note that this file uses "full_dataset".

### grid_search_optimization.py

3. With the virtual environment activated, run the program grid_search_optimization.py (it may take a moment):
    ```bash
    python3 grid_search_optimization.py
    ```
As a result, it saves the best model as "cnn_model_best.pth". Please note that this file uses "full_dataset".

## Step 5: Deactivate virtual environment
```bash
deactivate
```
---
