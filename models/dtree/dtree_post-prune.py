from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV


print('Loading train data...')
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
print('Finished loading train data')

tree = DecisionTreeClassifier(random_state=0, class_weight='balanced')
print('Evaluating cost complexity pruning path...')
path = tree.cost_complexity_pruning_path(train_data, train_labels)
print('Finished evaluating cost complexity pruning path')
hyperparams = {'ccp_alpha':path.ccp_alphas}
tree = RandomizedSearchCV(tree, hyperparams, scoring='f1', n_jobs=1, cv=5, random_state=0)

print('Fitting decision tree...')
tree = tree.fit(train_data, train_labels)
print('Finished fitting decsion tree')

print('Saving decision tree to cwd...')
joblib.dump(tree, 'dtree_post-prune.joblib')
print('Decision tree saved to cwd')
