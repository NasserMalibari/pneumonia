# COMP9417 Group Project

This project contains the code for the 2022 T2 COMP9417 Group Project for team Spaghetti Code. It tackles the machine learning challenge on [kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) revolving around the classification of X-rays of viral and bacterial pneumonia against healthy lungs. 

## Getting Started

These instructions will explain how to setup and run our different models to replicate our results.

### Installing
To download the dataset needed for this project visit this [kaggle project](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), create an account and download the data zip file. Unzip the compressed file and extract chest_xray into the base directory of this project so that you have the following file structure:

    pycache
    models
    chest_xray
    README.md

To install the packages needed for this project run:

    pip install {dependancy}

## Running the models

Instructions for running each of the models

### Neural Net

Run with working directory as the root directory.

Train the model by running:

    python3 models/neural_net/neural_net.py

Note this will take longer if running the first time to generate mirrored images, see 4.3

### Decision Tree

Run with working directory in '/models/dtree'.

First preprocess dataset to generate .npy files

    python3 dtree_pre.py

Then train each model by running:

    python3 dtree_{model_name}.py

This will generate the models which are .joblib files, to then be loaded and analysed using

    python3 dtree_analysis.py {model_name}


### Logistic Regression

Run with

    python3 models/logistic_regression/logistic_regression.py

### Support Vector Machine

Preprocess the data by running and generate the .csv files

    python3 models/svm/pre.py

To train all models:

    python3 models/svm/svm.py

To see the confusion matrix results:

    python3 models/svm/results.py