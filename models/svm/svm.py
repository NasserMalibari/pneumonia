import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from skimage.io import imread
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
"""
# Linear SVM
parameterGrid = {'C':[1, 10, 100, 1000]}
svc = svm.LinearSVC(max_iter = 10000)
model = GridSearchCV(svc, parameterGrid)
"""
x_test = pd.read_csv('test_data.csv').to_numpy()
y_test = pd.read_csv('test_labels.csv').to_numpy().ravel()
x_train = pd.read_csv('train_data.csv').to_numpy()
y_train = pd.read_csv('train_labels.csv').to_numpy().ravel()

sc = StandardScaler().fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
"""
# Fit the model
model.fit(x_train_std, y_train)
train_score = model.score(x_train_std, y_train)
print(f"train score: {train_score}")
test_score = model.score(x_test_std, y_test)
print(f"test_score: {test_score}")
print(f"best C chosen: {model.best_params_}")
bestC = model.best_params_['C']

# Model testing
y_pred = model.predict(x_test_std)
cm = metrics.confusion_matrix(y_test, y_pred)
cr = metrics.classification_report(y_test, y_pred)
acc = metrics.accuracy_score(y_test, y_pred)
print(cm)
print(cr)

print('-------- Change Penalty --------')

# Change penalty from l2 to l1 and changing C
parameterGridPenalty = {'penalty':['l1', 'l2']}
svcPenalty = svm.LinearSVC(max_iter = 10000, C = 1, dual = False)
modelPenalty = GridSearchCV(svcPenalty, parameterGridPenalty)
modelPenalty.fit(x_train_std, y_train)
train_score_penalty = modelPenalty.score(x_train_std, y_train)
print(f"train score: {train_score_penalty}")
test_score_penalty = modelPenalty.score(x_test_std, y_test)
print(f"test_score: {test_score_penalty}")
print(f"best penalty chosen: {modelPenalty.best_params_}")

y_pred_penalty = modelPenalty.predict(x_test_std)
cm_penalty = metrics.confusion_matrix(y_test, y_pred_penalty)
cr_penalty = metrics.classification_report(y_test, y_pred_penalty)
acc = metrics.accuracy_score(y_test, y_pred_penalty)
print(cm_penalty)
print(cr_penalty)

print('-------- Change Loss --------')

# Change loss from squared_hinge to hinge need to use default l2 penalty
parameterGridLoss = {'loss':['hinge', 'squared_hinge']}
svcLoss = svm.LinearSVC(max_iter = 10000, C = 1)
modelLoss = GridSearchCV(svcLoss, parameterGridLoss)
modelLoss.fit(x_train_std, y_train)
train_score_loss = modelLoss.score(x_train_std, y_train)
print(f"train score: {train_score_loss}")
test_score_loss = modelLoss.score(x_test_std, y_test)
print(f"test_score: {test_score_loss}")
print(f"best loss chosen: {modelLoss.best_params_}")

y_pred_loss = modelLoss.predict(x_test_std)
cm_loss = metrics.confusion_matrix(y_test, y_pred_loss)
cr_loss = metrics.classification_report(y_test, y_pred_loss)
print(cm_loss)
print(cr_loss)

print('-------- Change Loss & C --------')

# Cross validation with C and changing loss function
parameterFinal = {'C':[1, 10, 100, 1000], 'loss':['hinge', 'squared_hinge']}
svcFinal = svm.LinearSVC(max_iter = 10000)
modelFinal = GridSearchCV(svcFinal, parameterFinal)
modelFinal.fit(x_train_std, y_train)
train_score_final = modelFinal.score(x_train_std, y_train)
print(f"train score: {train_score_final}")
test_score_final = modelFinal.score(x_test_std, y_test)
print(f"test_score: {test_score_final}")
print(f"best loss chosen: {modelFinal.best_params_}")

y_pred_final = modelFinal.predict(x_test_std)
cm_final = metrics.confusion_matrix(y_test, y_pred_final)
cr_final = metrics.classification_report(y_test, y_pred_final)
print(cm_final)
print(cr_final)

print('-------- Non Linear Kernel --------')

# Non-linear kernel SVM
# Do not run without dimensionality reduction as it will take forever
param_grid = {'kernel':['rbf', 'poly']}
svc2 = svm.SVC(probability = False, gamma = 'auto')
model2 = GridSearchCV(svc2, param_grid)

model2.fit(x_train_std, y_train)
train_score2 = model2.score(x_train_std, y_train)
print(f"train score: {train_score2}")
test_score2 = model2.score(x_test_std, y_test)
print(f"test_score: {test_score2}")
print(f"best parameters: {model2.best_params_}")

# Model testing
y_pred2 = model2.predict(x_test_std)
cm_nonlinear = metrics.confusion_matrix(y_test, y_pred2)
cr_nonlinear = metrics.classification_report(y_test, y_pred2)
acc = metrics.accuracy_score(y_test, y_pred2)
print(cm_nonlinear)
print(cr_nonlinear)

print('--------- Non Linear with C and different kernels --------')

param_grid = {'C':[1, 10, 100], 'kernel':['rbf', 'poly']}
svc3 = svm.SVC(probability = False)
model3 = GridSearchCV(svc3, param_grid)

model3.fit(x_train_std, y_train)
train_score3 = model3.score(x_train_std, y_train)
print(f"train score: {train_score3}")
test_score3 = model3.score(x_test_std, y_test)
print(f"test_score: {test_score3}")
print(f"best parameters: {model3.best_params_}")

# Model testing
y_pred3 = model3.predict(x_test_std)
cm_nonlinear_2 = metrics.confusion_matrix(y_test, y_pred3)
cr_nonlinear_2 = metrics.classification_report(y_test, y_pred3)
acc = metrics.accuracy_score(y_test, y_pred3)
print(cm_nonlinear_2)
print(cr_nonlinear_2)
"""
print('-------- Non Linear Kernel Chosen Parameters --------')

# Non-linear kernel SVM
# Do not run without dimensionality reduction as it will take forever
param_grid = {'C':[0.1, 1, 10, 100]}
svc2 = svm.SVC(probability = False, gamma = 'auto', kernel = 'rbf')
model2 = GridSearchCV(svc2, param_grid)

model2.fit(x_train_std, y_train)
train_score2 = model2.score(x_train_std, y_train)
print(f"train score: {train_score2}")
test_score2 = model2.score(x_test_std, y_test)
print(f"test_score: {test_score2}")
print(f"best parameters: {svc2.best_params_}")

# Model testing
y_pred2 = model2.predict(x_test_std)
cm_nonlinear = metrics.confusion_matrix(y_test, y_pred2)
cr_nonlinear = metrics.classification_report(y_test, y_pred2)
acc = metrics.accuracy_score(y_test, y_pred2)
print(cm_nonlinear)
print(cr_nonlinear)