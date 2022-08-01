import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# https://towardsdatascience.com/random-forest-hyperparameters-and-how-to-fine-tune-them-17aee785ee0d

print('Loading train data...')
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
print('Finished loading train data')

tree = RandomForestClassifier(n_estimators=100, max_features='sqrt', criterion='gini', class_weight='balanced', n_jobs=-1, random_state=0)
hyperparams = {'max_depth':[4, 8, 12, 16, 20, 24]}
tree = GridSearchCV(tree, hyperparams, scoring='f1', n_jobs=-1, cv=5)

print('Fitting decision tree...')
tree = tree.fit(train_data, train_labels)
print('Finished fitting decsion tree')

print('Saving decision tree to cwd...')
joblib.dump(tree, 'dtree_rf.joblib')
print('Decision tree saved to cwd')