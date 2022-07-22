#Ensembling technique for the cocoa comapny data set
#importing the packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame

company_data = pd.read_excel("C:\\Users\\ADMIN\\OneDrive\\Desktop\\ensemble technique\\Coca_Rating_Ensemble.xlsx")
#Finding any missing values in the data set
company_data.isnull().sum()
company_data.dropna()

#Segregating the taxable income as good and risky
highratings = DataFrame(company_data.Rating)
highratings.loc[highratings['Rating']<=3,'highratings'] = 'Low'
highratings.loc[highratings['Rating']>3,'highratings'] = 'High'
highratings

#Creating dummy variable for the catagorical data
lb = LabelEncoder()
company_data["Company"] = lb.fit_transform(company_data["Company"])
company_data["Name"] = lb.fit_transform(company_data["Name"])
company_data["Company_Location"] = lb.fit_transform(company_data["Company_Location"])
company_data["Bean_Type"] = lb.fit_transform(company_data["Bean_Type"])
company_data["Origin"] = lb.fit_transform(company_data["Origin"])

highratings['highratings'].unique()
highratings['highratings'].value_counts()

#Segregating the data set as output and input data sets
predictors = company_data.drop(["highratings"], axis = 1)
target = highratings.drop(["highratings"], axis = 1)

#Splitting the data into taining and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state=0)

#Voting model for the data set
from sklearn import linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

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


#Applying the bagging model

from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


bagging_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bagging_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bagging_clf.predict(x_test))
accuracy_score(y_test, bagging_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_test, bagging_clf.predict(x_train))
accuracy_score(y_test, bagging_clf.predict(x_train))

#Boosting models on the data set
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))

#Gradient boosting for the data set

from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, boost_clf.predict(x_test))
accuracy_score(y_test, boost_clf.predict(x_test))

