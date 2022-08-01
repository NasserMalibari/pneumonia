from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV


print('Loading train data...')
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
print('Finished loading train data')

tree = DecisionTreeClassifier(random_state=0, class_weight='balanced')
hyperparams = {'min_samples_leaf':[1, 2, 4, 8, 16, 32], 'min_samples_split':[2, 4, 8, 16, 32], 'max_depth':[1, 2, 4, 8, 16, 32, None]}
tree = RandomizedSearchCV(tree, hyperparams, scoring='f1', n_jobs=3, cv=5, random_state=0)

print('Fitting decision tree...')
tree = tree.fit(train_data, train_labels)
print('Finished fitting decsion tree')

print('Saving decision tree to cwd...')
joblib.dump(tree, 'dtree_pre-prune.joblib')
print('Decision tree saved to cwd')
