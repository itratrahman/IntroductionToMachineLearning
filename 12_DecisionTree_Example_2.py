####This source gives a demonstraion of decision tree classifier and cross validation 

##import statements
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

irisData = pd.read_csv("E:\Introduction to Machine Learning\iris_data.csv")

##Printing the first five data of the set
# print irisData.head()
# print '\n'
##Print out the description of the data features 
# print irisData.describe()
# print '\n'
##Print out the correlation matrix
# print irisData.corr()
# print "\n"

##Extracting the feature vector from the data set
features = irisData[["SepalLength", "SepalWidth","PetalLength","PetalWidth"]]
# print features

##Extracting the target variable from the data set
targetVariables = irisData.Class

##Splitting the data set for cross validation with tast data size of 20%
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size = 0.2)

##Instantiating the decision tree classifier
model = DecisionTreeClassifier()

##Training the training set using the classifier
fittedModel = model.fit(featureTrain, targetTrain)

##Creating the prediction vector for the fitted model
predictions = fittedModel.predict(featureTest)
# print predictions

##Printing out the confusion_matrix fo the prediction using the classifier
print confusion_matrix(targetTest, predictions)

print "\n"

##Printing out the accruacy of the prediction using the classifier
print accuracy_score(targetTest, predictions)
