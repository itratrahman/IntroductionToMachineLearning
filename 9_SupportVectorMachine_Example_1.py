####This sourcecode gives a simple demonstration of support machines

##import statements for numpy, matplotlib, and sklearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

####Setting the training data for plotting
##Setting the x and y coordinates of blue color data
xBlue = np.array([0.3,0.5,1,1.4,1.7,2])
yBlue = np.array([1,4.5,2.3,1.9,8.9,4.1])
##Setting the x and y coordinates of red color data
xRed = np.array([3.3,3.5,4,4.4,5.7,6])
yRed = np.array([7,1.5,6.3,1.9,2.9,7.1])
##Plotting the data points for different classes
plt.plot(xBlue,yBlue,'ro', color = 'blue')
plt.plot(xRed,yRed,'ro', color = 'red')

##Setting the training dataset 
x = np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1])

##Cordinates chosen for prediction using classifier
xNew1 = 3 
xNew2 = 4 

##Instantiating a support vector machine classifier
classifier = svm.SVC()

##Training the data with the classifier
classifier.fit(x,y)

##Printing out the prediction using the trained classifier
print "Prediction: "
print(classifier.predict([xNew1,xNew2]))

##Plotting the chose data point 
plt.plot(xNew1,xNew2,'ro',color='green', markersize = 15)

##Setting the axis
plt.axis([-0.5,10,-0.5,10])

##Displaying the grids
plt.grid(True)

##Displaying the plots
plt.show()