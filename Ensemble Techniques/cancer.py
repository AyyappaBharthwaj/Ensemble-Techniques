####################### Bagging #####################################
import pandas as pd

df = pd.read_csv("E:/Data Science 18012022/Ensemble Techniques/Tumor_Ensemble.csv")

# Dummy variables
df.head()
df.info()

df = pd.get_dummies(df, columns = ["diagnosis"], drop_first = True)
df.head()
# Input and Output Split
predictors = df.loc[:, df.columns!=" diagnosis_M"]
type(predictors)

target = df["diagnosis_M"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bag_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))

######################################boosting################################################
#pip install xgboost
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

# n_jobs – Number of parallel threads used to run xgboost.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)


xgb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, cv_xg_clf.predict(x_test))
grid_search.best_params_

##########################stacking########################################
# Libraries and data loading

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np



# Create the ensemble's base learners and meta learner
# Append base learners to a list
base_learners = []

# KNN classifier model
knn = KNeighborsClassifier(n_neighbors=2)
base_learners.append(knn)

# Decision Tree Classifier model
dtr = DecisionTreeClassifier(max_depth=4, random_state=123456)
base_learners.append(dtr)

# Multi Layered Perceptron classifier
mlpc = MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456)
base_learners.append(mlpc)

# Meta model using Logistic Regression
meta_learner = LogisticRegression(solver='lbfgs')


# Create the training meta data

# Create variables to store meta data and the targets
meta_data = np.zeros((len(base_learners), len(x_train)))
meta_targets = np.zeros(len(x_train))

# Create the cross-validation folds
KF = KFold(n_splits = 5)
meta_index = 0
for train_indices, test_indices in KF.split(x_train):
    # Train each learner on the K-1 folds and create meta data for the Kth fold
    for i in range(len(base_learners)):
        learner = base_learners[i]

        learner.fit(x_train[train_indices], y_train[train_indices])
        predictions = learner.predict_proba(x_train[test_indices])[:,0]

        meta_data[i][meta_index:meta_index+len(test_indices)] = predictions

    meta_targets[meta_index:meta_index+len(test_indices)] = y_train[test_indices]
    meta_index += len(test_indices)

# Transpose the meta data to be fed into the meta learner
meta_data = meta_data.transpose()

# Create the meta data for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(x_test)))
base_acc = []

for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(x_train, y_train)
    predictions = learner.predict_proba(x_test)[:,0]
    test_meta_data[i] = predictions

    acc = metrics.accuracy_score(y_test, learner.predict(x_test))
    base_acc.append(acc)
    
test_meta_data = test_meta_data.transpose()

# Fit the meta learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data, meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)

acc = metrics.accuracy_score(y_test, ensemble_predictions)

# Print the results
for i in range(len(base_learners)):
    learner = base_learners[i]

    print(f'{base_acc[i]:.2f} {learner.__class__.__name__}')
    
print(f'{acc:.2f} Ensemble')

#########################voting#################################
# Import the required libraries
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(x_train, y_train)

# Predict the most voted class
hard_predictions = voting.predict(x_test)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions))

#################

# Soft Voting # 
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(x_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(x_test)
predictions_5 = learner_5.predict(x_test)
predictions_6 = learner_6.predict(x_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))



