from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib


print('Loading train data...')
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
print('Finished loading train data')

tree = DecisionTreeClassifier(random_state=0, class_weight='balanced')

print('Fitting decision tree...')
tree = tree.fit(train_data, train_labels)
print('Finished fitting decision tree')

print('Saving decision tree to cwd...')
joblib.dump(tree, 'dtree_base.joblib')
print('Decision tree saved to cwd')
