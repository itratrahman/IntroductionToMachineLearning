####This source code demonstrates simple logittic regression with multiple features as input

##import statements for numpy, matplotlib, sklearn.linear_model
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

##Logistic Regression Model
#pi = 1 / (1 + exp[-(b0 + b1 *x)])

##format of the x training data: balance; income; age]

##trainig data set
#x data set with 3 features
x = np.array([[10000,80000,35],[7000,120000,57],[100,23000,22],[223,18000,26]])
#y data set with 2 different classes
y = np.array([1,1,0,0])

##Instantiating a logistic regression classifier
classifier = LogisticRegression()

##Training the data set with the logistic regression classifier
classifier.fit(x,y)

###Retrieving the outcome with a specific data set using the data trained via logistic regression:
#balance: 5500; income: 50000, age: 25
print(classifier.predict([5500,50000,25]))