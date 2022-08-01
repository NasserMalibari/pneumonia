from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier

print('Loading train data...')
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
print('Finished loading train data')

tree = DecisionTreeClassifier(random_state=0, class_weight='balanced')
# What does class_weight do? Solution to imbalanced dataset (number of pneumonia positive cases >>> negative cases)
# Weight the classes by the inverse of their sample size, so that the model considers both classes equally on the split criteria
# https://datascience.stackexchange.com/questions/56250/how-does-class-weight-work-in-decision-tree
# https://stackoverflow.com/questions/37522191/how-to-balance-classification-using-decisiontreeclassifier

# Why is there randomness in decision trees?
# If we limit the number of features to consider in each split, then the selected features to consider will be random
# Even if don't limit the number of features, if the improvement of the criterion is identical for several splits then one split has to be selected at random.
# https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn


"""
Strategies to prevent overfitting (or also improve bias in ensemble learning)
https://towardsdatascience.com/3-techniques-to-avoid-overfitting-of-decision-trees-1e7d3d985a09


Pre-pruning:
The hyperparameters of the decision tree including max_depth, min_samples_leaf, min_samples_split
(clarification on min_samples_leaf vs min_samples_split)
https://stackoverflow.com/questions/46480457/difference-between-min-samples-split-and-min-samples-leaf-in-sklearn-decisiontre
How to select hyperparameters?
Choosing an optimization strategy
https://towardsdatascience.com/machine-learning-algorithms-and-the-art-of-hyperparameter-selection-279d3b04c281
What is cross validation? With only training data, we can measure how accurate a model is on unseen data
(meaning we measure accuracy taking into account overfitting, when normally we can't)
https://stackoverflow.com/questions/2314850/help-understanding-cross-validation-and-decision-trees
Therefore, we can use cross validation to find the optimal hyperparameters for the model that best prevents overfitting
So how do we do it?
https://stackoverflow.com/questions/35097003/cross-validation-decision-trees-in-sklearn


Post-pruning:
Cost complexity pruning (ccp), involves tuning ccp_alpha to get the best fit model

Ensemble learning:
Random forest
Boosting techniques (Adaboost, gradient boost)
Bagging vs Boosting??
In many cases, bagging methods constitute a very simple way to improve with respect to a single model, 
without making it necessary to adapt the underlying base algorithm. 
As they provide a way to reduce overfitting, bagging methods work best with strong and complex models 
(e.g., fully developed decision trees), in contrast with boosting methods which usually work 
best with weak models (e.g., shallow decision trees).

Which optimisation strategy should I use?
In terms of producing better results: Ensemble learning > Post-pruning > Pre-pruning
In terms of running faster: Pre-pruning > Post-pruning > Ensemble learning
https://towardsdatascience.com/pre-pruning-or-post-pruning-1dbc8be5cb14
"""

""" PRE-PRUNING """
hyperparams = {'min_samples_leaf':[1, 2, 4, 8, 16], 'min_samples_split':[2, 4, 8, 16, 32], 'max_depth':[1, 2, 4, 8, 16, 32]}

# Note that min_samples_split > min_samples_leaf to be meaningful

tree = RandomizedSearchCV(tree, hyperparams, scoring='f1', n_jobs=4, cv=5, random_state=0)

""" ENSEMBLE LEARNING: random forest """
# tree = RandomForestClassifier(n_estimators=500, class_weight='balanced', max_features='sqrt', n_jobs=4, random_state=0)

""" ENSEMBLE LEARNING: extremely random forest """
# tree = ExtraTreesClassifier(n_estimators=500, class_weight='balanced', max_features='sqrt', n_jobs=4, random_state=0)

""" ENSEMBLE LEARNING: adaboost """
# tree = AdaBoostClassifier(base_estimator=tree, n_estimators=100, learning_rate=1, random_state=0)

""" ENSEMBLE LEARNING: gradient boosting """
# tree = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=3, random_state=0)

print('Fitting decision tree...')
tree = tree.fit(train_data, train_labels)
print('Finished fitting decsion tree')

print('Saving decision tree to cwd...')
joblib.dump(tree, 'dtree.joblib')
print('Decision tree saved to cwd')
