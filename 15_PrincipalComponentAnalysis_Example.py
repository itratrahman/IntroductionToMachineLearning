####This source code gives a demonstration of Principal Componet Analysis with a dataset containing 10 digits

##import statements
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

# 8x8 pixel per image = 64 features | Humans are not able to cope with this amount of dimensionality
# This is the reason for using PCA; it reduces the dimensions: we can visualize the data in 2D
# We want to investigate if the distribution  after PCA reveals the distribution
# of the different classed, and if they are clearly seperable

##Extracting the datasets for 8x8 digit
digits = load_digits()

##Extracting the training set(feature vector, targetvaraibles) from the dataset 
X_digits, Y_digits = digits.data, digits.target

##Parameter for figure shape and size
n_row, n_col, max_n = 2, 5, 10

##Shaping the size of the figure
fig = plt.figure(figsize=(2. * n_col, 2.26 * n_row))

##While loop counter variable
i = 0

##While loop for plotting the subplot (image of the digit) of the figure
while i<max_n and i<digits.images.shape[0]:
	
	##adding a subplot
	p = fig.add_subplot(n_row, n_col, i + 1, xticks =[], yticks = [])
	##Displaying the digit image with the bone colormap on the current subplot
	p.imshow(digits.images[i], cmap = plt.cm.bone, interpolation = 'nearest')
	##giving label to the subplot
	p.text(0,-1,str(digits.target[1]))
	##incrementing the counter
	i = i + 1
	
##Plot show command
plt.show()

##Varaible to store the number of components of PCA
numberOfComponents = 10

#Instantiating the Principal Component Analysis Object with given number of components
estimator = PCA(n_components = numberOfComponents)

##Performing PCA on the feature dataset
X_pca = estimator.fit_transform(X_digits)

##Setting the color for the components of the PCA 
colors = ['black', 'blue', 'purple', 'yellow', 'white','red', 'lime', 'cyan', 'orange', 'gray']

for i in range(len(colors)):

	##Extracting the x and y data of the current (ith) components
	px = X_pca[:,0][Y_digits == i]
	py = X_pca[:,1][Y_digits == i]
	##Plotting a scatter plot with the extracted x & y data for the selected(ith) color
	plt.scatter(px,py, c = colors[i])
	##Setting the legend of the plot
	plt.legend(digits.target_names)
	##Setting the axis label
	plt.xlabel("First Principal Component")
	plt.ylabel("Second Principal Component")
	
plt.show()
	
	


