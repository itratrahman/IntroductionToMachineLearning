####This source code gives a simple demonstration of KNN classifier with muliple class

##import statements
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

##data for Fruits
xFruit = np.array([10,10])
yFruit = np.array([9,1])

##Coordinates for Proteins
xProtein = np.array([1,1])
yProtein = np.array([4,1])

##Coordinates for Vegetables
xVegetable = np.array([7])
yVegetable = np.array([10])

##Training Dataset
x = np.array([[10,9],[10,1],[1,4],[1,1],[7,10]])
y = np.array([0,0,1,1,2]) # 0: FRUIT, 1: PROTEIN, 2: VEGETABLE

##Plotting the coordinates of the Ingredients
plt.plot(xFruit,yFruit,'ro',color = 'blue')
plt.plot(xProtein,yProtein,'ro',color = 'green')
plt.plot(xVegetable,yVegetable,'ro',color = 'yellow')

##Corrdinates of the chosen for to test prediction
x_axis = 6
y_axis = 4

##Plotting the chosen point
plt.plot(x_axis, y_axis, 'ro', color = 'gray', markersize = 15)

##instantiating a kNN classifier
k = 3 #number of k neighbors
classifier = KNeighborsClassifier(n_neighbors = k)

##Training the data set using the KNN classifier  
classifier.fit(x,y)

##predicting using the classifier predictor object with the chosen points
value = classifier.predict([x_axis,y_axis]) 
print value

##Fixing the axis
plt.axis([-0.5,15,-0.5,15])

##displaying the grids
plt.grid(True)

##Displaying the plots
plt.show()

