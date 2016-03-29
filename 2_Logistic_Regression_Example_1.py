####This source code gives a simple demonstation of logistic regression

##import statements for numpy, matplotlib, and sklearn.linear_model
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

##Logistic Regression Model
#pi = 1 / (1 + exp[-(b0 + b1 *x)])

####Creating two classes of dataset for plotting purpose
##data set having positive class 
x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
##data set having negative class
x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

####This Cource code gives a simple demonstation of Logistic Regression

##plotting the data sets
plt.plot(x1,y1,'ro', color = 'blue')
plt.plot(x2,y2,'ro', color = 'red')

####Training dataset for the two classes
##complete x data set
x = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],[3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]]) 
##Complete y data set
y = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

##Instantiating a logistic regression classifier
classifier = LogisticRegression()

##Training the dataset with logistic regression classifier
classifier.fit(x,y)

##Creating a function for calculating the probality using the sigmoid function
#passing the classifier object and the x data as arguments to the function
def model(classifier, x):
	return 1/(1 + np.exp(-(classifier.intercept_ + classifier.coef_ * x)))
	
##plot the logistic regression in the window point by point in a for loop
#using 120 data points passed into the our created function "model"
for i in range(1,120,1):
	plt.plot(i/10.0-2, model(classifier,i/10.0-2),'ro', color = 'green')

#method1 of calculating probality using a data: using the builtin method
value = classifier.predict_proba(8)
print(value)

#method2 of calculating probality using a data: using our created function
print(model(classifier,8))
 
##Setting the axis
plt.axis([-2, 10, -0.5, 2])

##displaying the grids
plt.grid(True)

##displaying the window
plt.show() 