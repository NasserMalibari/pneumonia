# COMP9417 Group Project

This project contains the code for the 2022 T2 COMP9417 Group Project for team Spaghetti Code. It tackles the machine learning challenge on [kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) revolving around the classification of X-rays of viral and bacterial pneumonia against healthy lungs. 

## Getting Started

These instructions will explain how to setup and run our different models to replicate our results.

### Installing

    pip install {dependancy}

## Running the models

Instructions for running each of the models

### Neural Net

### Decision Tree

Run with working directory in '/models/dtree'.

First preprocess dataset to generate .npy files

    python3 dtree_pre.py

Then train each model by running:

    python3 dtree_{model_name}.py

This will generate the models which are .joblib files, to then be loaded and analysed using

    python3 dtree_analysis.py {model_name}


### Logistic Regression

Run with working directory in '/models/logistic_regression'.

    python3 logistic_regression.py

### Support Vector Machine

Preprocess the data by running and generate the .csv files

    python3 models/svm/pre.py

To train all models:

    python3 models/svm/svm.py

To see the confusion matrix results:

    python3 models/svm/results.py