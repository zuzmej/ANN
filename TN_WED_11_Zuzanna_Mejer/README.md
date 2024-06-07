# Image classification using CNN
Artificial Neural Network Project


# Description of content

In this directory there are several files:

* main.py  -- this is the main file, which does the following:
    + describe the CNN model,
    + train, evaluate and test the model,
    + plot validation and training loss graphs, validation accuracy graph and confusion matrix for test set.

* grid_search_optimization.py -- this file:
    + performs grid search,
    + chooses the best hyperparameters to the model,
    + tests the best hyperparameters.

* cnn_model.pth:
    + this is the file with saved trained model.

* test.py -- this is the file for testing:
    + use this file to test the trained saved model on exemplary dataset,
    + after finishing, it plots confusion matrix and prints accuracy,
    + follow undermentioned instructions to run the test file.

* dataset -- this is the directory with test set data, which are used in test file. It consists of 10 randomly chosen images of cats and 10 images of dogs. 

# Project setup instructions

## Step 1: Unpack the package

1. Unzip the package into a chosen directory.

## Step 2: Create and activate virtual environment

### On Windows:

1. Open Command Prompt and navigate to the project directory.
2. Create a virtual environment:
    ```bash
    python3 -m venv myenv
    ```
3. Activate the virtual environment:
    ```bash
    myenv\Scripts\activate
    ```

### On Linux/MacOS:

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
2. Install the required libraries using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Step 4: Run the program

1. With the virtual environment activated, run the program:
    ```bash
    python3 test.py
    ```

## Step 5: Deactivate virtual environment
```bash
deactivate
```
---


## Additional
If you would like to run main.py or grid_search_optimization.py files, you need to have dataset with 3 directories: training_set, validation_set, test_set. In each there should be 2 directories: cats and dogs with labeled images of these animals.