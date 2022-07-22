#Ensembling technique for the tumour data set
# Importing the packages
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Loading the data set into Python
breast_cancer = datasets.load_Tumor_Ensemble()
x, y = load_Tumor_Ensemble.data,load_Tumor_Ensemble.target

# Splitting the data into training and testing data 
test_samples = 500
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

# Instantiating the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiating the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(x_train, y_train)

# Predict the most voted class
hard_predictions = voting.predict(x_test)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions))

# Soft Voting # 
# Instantiaing the learners 
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


#Applying bagging model for the data set
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Importing the data set into python
tumour_data = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\ensemble technique\\Tumor_Ensemble.csv")

#Finding any missing values in the data set
tumour_data.isnull().sum()
tumour_data.dropna()

lb = LabelEncoder()
tumour_data["diagnosis"] = lb.fit_transform(tumour_data["diagnosis"])
tumour_data['diagnosis'].unique()
tumour_data['diagnosis'].value_counts()
colnames = list(tumour_data.columns)
predictors = colnames[:8]
target = colnames[8]


# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(tumour_data, test_size = 0.3)

#Applying the bagging model

from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


bagging_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bagging_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bagging_clf.predict(test[predictors]))
accuracy_score(y_test, bagging_clf.predict(test[predictors]))

# Evaluation on Training Data
confusion_matrix(y_test, bagging_clf.predict(train[predictors]))
accuracy_score(y_test, bagging_clf.predict(train[predictors]))

#Boosting models on the data set
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(test[predictors]))
accuracy_score(y_test, ada_clf.predict(test[predictors]))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(train[predictors]))

#Gradient boosting for the data set

from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, boost_clf.predict(test[predictors]))
accuracy_score(y_test, boost_clf.predict(test[predictors]))
