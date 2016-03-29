####This sourcecode gives a demonstration of croo validation performed on Logistic Regression Classifier

##import statements
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

##Reading the csv data with panda read method
creditData = pd.read_csv('E:\Introduction to Machine Learning\credit_data.csv')

####Displaying the relevant information contained within the rows of the csv file
##print the first 5 rows of the data
print creditData.head()
print("\n")
##Printing the statistical information of the data using the function describe
print creditData.describe()
print("\n")
##Print out the correlation between the data set
print creditData.corr()
print("\n")

##Extracting the features from the creditData
features = creditData[["income", "age", "loan"]]

##Extracting the target variables from the creditData
targetVariables = creditData.default

##Splitting the data set for cross validation with a test size of 20%
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size = 0.2);

##Instantiating a logistic regression classifier
model = LogisticRegression()

##Training the data set using logistic regression classifier
fittedmodel = model.fit(featureTrain, targetTrain)

##Performing cross validation using the pair of test data sets
predictions = fittedmodel.predict(featureTest)

print "the confusion matrix:" + "\n"
##Print the result of cross_validation using the confusion matrix
print confusion_matrix(targetTest, predictions)
print("\n")

print "the accuracy percentage of the cross validation result:" + "\n"
##Print the acuracy of cross_validation 
print accuracy_score(targetTest, predictions)
print("\n")
