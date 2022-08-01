import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

base_dir = '../../chest_xray/'

train_dir = base_dir + 'train/'
test_dir = base_dir + 'test/'
val_dir = base_dir + 'val/'

train_neg = train_dir + 'NORMAL'
train_pos = train_dir + 'PNEUMONIA'
test_neg = test_dir + 'NORMAL'
test_pos = test_dir + 'PNEUMONIA'
val_neg = val_dir + 'NORMAL'
val_pos = val_dir + 'PNEUMONIA'

train_pos = [train_pos+'/'+i  for i in os.listdir(train_pos) ]
train_neg = [train_neg + '/' + i for i in os.listdir(train_neg) ]


test_pos = [test_pos + '/' + i for i in os.listdir(test_pos) ]
test_neg = [test_neg + '/' + i for i in os.listdir(test_neg)]

val_pos = [val_pos + '/' + i for i in os.listdir(val_pos)]
val_neg = [val_neg + '/' + i for i in os.listdir(val_neg)]
print('---------------------------------------------------')

train_full = train_pos + train_neg

# size of smallest image()
image_size = 127

widths_train = []
heights_train = []

train_data = []
train_labels = []

#pre-process training images
count = 0
for train_img in train_full:
    img = cv2.imread(train_img, cv2.IMREAD_GRAYSCALE)
    
    widths_train.append(img.shape[0])
    heights_train.append(img.shape[1])
    
    img = cv2.resize(img, (image_size, image_size)).flatten()
    np_img = np.asarray(img)
    
    train_data.append(np_img)
    
    if "bacteria" in train_img or "virus" in train_img:
        train_labels.append(1)
    else:
        train_labels.append(0)

    if count % 750 == 0:
        print(f"{count} images processed")
    count += 1

test_data = []
test_labels = []

test_full = test_neg + test_pos


widths = []
heights = []

#pre-process test images
count = 0
for test_img in test_full:
    img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
    widths.append(img.shape[0])
    heights.append(img.shape[1])
    img = cv2.resize(img, (image_size, image_size)).flatten()
    np_img = np.asarray(img)
    test_data.append(np_img)
    if "bacteria" in test_img or "virus" in test_img:
        test_labels.append(1)
    else:
        test_labels.append(0)

    if count % 100 == 0:
        print(f"{count} images processed")
    count += 1


train_scores = []
test_scores = []

# convert to np arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)


test_data = np.array(test_data)
test_labels = np.array(test_labels)

#performs logistic regression using model thats passed in. Produces a confusion matrix as well as classification report to display f1 scores
def performLogisticRegression(logisticRegr):
  logisticRegr.fit(train_data , train_labels)
  y_pred_train = logisticRegr.predict(train_data)
  y_pred_test = logisticRegr.predict(test_data)

  cm = confusion_matrix(train_labels, y_pred_train)

  #plot confusion matrix on a heatmap
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.imshow(cm)
  ax.grid(False)
  ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
  ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
  ax.set_ylim(1.5, -0.5)
  for i in range(2):
      for j in range(2):
          ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
  plt.show()
  print(classification_report(train_labels, y_pred_train))
  
  cm = confusion_matrix(test_labels, y_pred_test)

  fig, ax = plt.subplots(figsize=(8, 8))
  ax.imshow(cm)
  ax.grid(False)
  ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
  ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
  ax.set_ylim(1.5, -0.5)
  for i in range(2):
      for j in range(2):
          ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
  plt.show()

  print(classification_report(test_labels, y_pred_test))

#Saga
performLogisticRegression(LogisticRegression(penalty="l1", tol=0.01, solver="saga"))
performLogisticRegression(LogisticRegression(penalty="l2", tol=0.01, solver="saga"))
performLogisticRegression(LogisticRegression(penalty="elasticnet", l1_ratio=0.5, tol=0.01, solver="saga"))
performLogisticRegression(LogisticRegression(penalty="l1", tol=0.01, solver="saga", class_weight="balanced"))
performLogisticRegression(LogisticRegression(penalty="l2", tol=0.01, solver="saga", class_weight="balanced"))
performLogisticRegression(LogisticRegression(penalty="elasticnet", l1_ratio=0.5, tol=0.01, solver="saga", class_weight="balanced"))

#Newton-cg
performLogisticRegression(LogisticRegression(penalty="l2", tol=0.01, solver="newton-cg"))
performLogisticRegression(LogisticRegression(penalty="none", tol=0.01, solver="newton-cg"))
performLogisticRegression(LogisticRegression(penalty="l2", tol=0.01, solver="newton-cg", class_weight="balanced"))
performLogisticRegression(LogisticRegression(penalty="none", tol=0.01, solver="newton-cg", class_weight="balanced"))

#Sag
performLogisticRegression(LogisticRegression(penalty="l2", tol=0.01, solver="sag"))
performLogisticRegression(LogisticRegression(penalty="none", tol=0.01, solver="sag"))
performLogisticRegression(LogisticRegression(penalty="l2", tol=0.01, solver="sag", class_weight="balanced"))
performLogisticRegression(LogisticRegression(penalty="none", tol=0.01, solver="sag", class_weight="balanced"))

#lbfgs
performLogisticRegression(LogisticRegression(penalty="l2", tol=0.01, solver="lbfgs"))
performLogisticRegression(LogisticRegression(penalty="none", tol=0.01, solver="lbfgs"))
performLogisticRegression(LogisticRegression(penalty="l2", tol=0.01, solver="lbfgs", class_weight="balanced"))
performLogisticRegression(LogisticRegression(penalty="none", tol=0.01, solver="lbfgs", class_weight="balanced"))