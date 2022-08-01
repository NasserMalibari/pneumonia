import sys
from tkinter.tix import Tree
from sklearn import metrics
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
# Compare one classifier’s overall performance to another in a single metric — use Matthew’s correlation coefficient, Cohen’s kappa, and log loss.
# Measure a classifier’s ability to differentiate between each class in balanced classification: ROC AUC score
# A metric that minimizes false positives and false negatives in imbalanced classification: F1 score
# Focus on decreasing the false positives of a single class: Precision for that class
# Focus on decreasing the false negatives of a single class: Recall for that class.

if len(sys.argv) != 2:
    print(f'error {sys.argv[0]}: require model name', file=sys.stderr)
    sys.exit(1)

modelname = sys.argv[1]

print('Loading test data...')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')
print('Finished loading test data')

print('Loading trained decision tree...')
tree = joblib.load(modelname+'.joblib')
print('Finished loading trained decision tree')

predictions = tree.predict(test_data)

cm = metrics.confusion_matrix(test_labels, predictions)
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True, xticklabels=['Predicted NO', 'Predicted YES'], yticklabels=['Actual NO', 'Actual YES'])
plt.title(f'Confusion matrix for {modelname}')
plt.savefig(modelname+'_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.show()

print('--- Confusion matrix ---')
print(f"""         Predicted NO    |   Predicted YES
Actual NO       {cm[0][0]}              {cm[0][1]}
Actual YES      {cm[1][0]}              {cm[1][1]}""")

f1_score = metrics.f1_score(test_labels, predictions)

print('--- F1 score ---')
print(f1_score)

precision_score = metrics.precision_score(test_labels, predictions)

print('--- Precision score ---')
print(precision_score)

recall_score = metrics.recall_score(test_labels, predictions)

print('--- Recall score ---')
print(recall_score)

if hasattr(tree, 'best_params_'):
    print('--- Best params ---')
    print(tree.best_params_)

if hasattr(tree, 'best_estimator_') and hasattr(tree.best_estimator_, 'feature_importances_'):
    feature_reshaped = tree.best_estimator_.feature_importances_.reshape(127, 127)
    plt.matshow(feature_reshaped, cmap=plt.cm.hot)
    plt.title(f'Pixel importances of {modelname} using impurity values')
    plt.colorbar()
    plt.savefig(modelname+'_feature_importance_heatmap.png', dpi=100, bbox_inches='tight')
    plt.show()
elif hasattr(tree, 'feature_importances_'):
    feature_reshaped = tree.feature_importances_.reshape(127, 127)
    plt.matshow(feature_reshaped, cmap=plt.cm.hot)
    plt.title(f'Pixel importances of {modelname} using impurity values')
    plt.colorbar()
    plt.savefig(modelname+'_feature_importance_heatmap.png', dpi=100, bbox_inches='tight')
    plt.show()