from xgboost import XGBClassifier
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import multiprocessing

print('Loading train data...')
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
print('Finished loading train data')

neg_over_pos = (train_labels.size - np.sum(train_labels)) / np.sum(train_labels)
tree = XGBClassifier(n_estimators=100, scale_pos_weight=neg_over_pos, random_state=0, tree_method='gpu_hist')
hyperparams = {'max_depth':[8, 16, 24], 'subsample':[0.1, 0.5, 1.0]}
tree = GridSearchCV(tree, hyperparams, scoring='f1', n_jobs=4, cv=5)

print('Fitting decision tree...')
tree = tree.fit(train_data, train_labels)
print('Finished fitting decsion tree')

print('Saving decision tree to cwd...')
joblib.dump(tree, 'dtree_gb.joblib')
print('Decision tree saved to cwd')