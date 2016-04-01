####This sourcecode gives a demonstration of random forest classifier and cross validation method

##import statements
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

##Reading the csv data with panda read method
creditData = pd.read_csv('E:\Introduction to Machine Learning\credit_data.csv')

##Printing the first five data of the set
print creditData.head()
print '\n'
##Print out the description of the data features 
print creditData.describe()
print '\n'
##Print out the correlation matrix
print creditData.corr()
print "\n"

##Extracting the feature vector from the dataset
features = creditData[["income", "age", "loan"]]
# print features 
# print "\n"

##Extracting the target variable from the dataset
targetVariables = creditData.default

##Splitting the data set for cross validation
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size = 0.2)

##Instantiating the random forest classifier with a given number of estimators/no. of random forest
model = RandomForestClassifier(n_estimators = 25)

##Training the data with the classifier
fittedModel = model.fit(featureTrain, targetTrain)

##Creating the prediction vector featureTest
predictions = fittedModel.predict(featureTest)

##Print out the result of the predictions using the confusion matrix
print confusion_matrix(targetTest, predictions)
##Print out the result of the predictions using the confusion matrix
print accuracy_score(targetTest,predictions)
