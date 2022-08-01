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

# Linear SVM 1
cm = np.array([[93, 141], [10, 379]])
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Linear SVM 2
cm = np.array([[84, 150], [2, 387]])
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Linear SVM 3
cm = np.array([[93, 141], [10, 379]])
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Linear SVM 4
cm = np.array([[93, 141], [10, 379]])
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Non-linear SVM 1
cm = np.array([[102, 132], [5, 384]])
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Non-linear SVM 2
cm = np.array([[91, 143], [4, 385]])
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
