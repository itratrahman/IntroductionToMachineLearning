####This source code gives a simple demonstration of KNN classifier

##import statements
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

##Setting the x and y coordinates of blue color data
xBlue = np.array([0.3,0.5,1,1.4,1.7,2])
yBlue = np.array([1,4.5,2.3,1.9,8.9,4.1])

##Setting the x and y coordinates of red color data
xRed = np.array([3.3,3.5,4,4.4,5.7,6])
yRed = np.array([7,1.5,6.3,1.9,2.9,7.1])

##Setting the training dataset 
x = np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1]) # 0: blue class; 1: red class

##Plotting the red and the blue data point on the plot
plt.plot(xBlue,yBlue,'ro', color = 'blue')
plt.plot(xRed,yRed,'ro', color = 'red')

##Instantiating a KNN classifier
classifier = KNeighborsClassifier(n_neighbors = 3)

##Training the data set using the KNN classifier
classifier.fit(x,y)

##Data point for predection
newX1 = 2
newX2= 5

##Plotting the datapoint
plt.plot(newX1,newX2, 'ro',color = 'green', markersize = 15)

##Giving a prediction using a training point
pred = classifier.predict([newX1,newX2])

##Printing the classifier prediction
print "prediction: "
print(pred)

##Setting the axis
plt.axis([-0.5,10,-0.5,10])

##Displaying the grids
plt.grid(True)

##Displaying the plots
plt.show()
