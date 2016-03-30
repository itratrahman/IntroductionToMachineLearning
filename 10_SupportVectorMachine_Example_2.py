####This source code gives a demonstration of character recognition using support vector machine

##Import Statements for matplotlib and sklean libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets
from sklearn import svm

##Storing the dataset containing the images
numberImages = datasets.load_digits()

##Print the shape of the datasets
# print numberImages.data.shape

####Visualizing the grayscale image using the matrix dataset
# imageNumber = 10 #image from the dataset
# plt.matshow(numberImages.images[imageNumber],cmap=cm.Greys_r) 
# plt.show()

##Instantiating the support vector machine classifier
model = svm.SVC(gamma = 0.0001, C = 100, kernel = 'rbf') ##C perimeter is the error term, if it is too large 
														 #then we have a high penanlty for non-seperable points, and we may overfit
														 #and if it is too small we may underfit
														 ##gamma is called the kernel coefficient for the radial basis function
														 #which controls how far the influence of a single training example reaches
														 #if it is very low then it means that these are far away; if it is high then they are close
							
##Extracting the feature set and target varaibles and storing them in appropriate variables 
x, y = numberImages.data[:-5], numberImages.target[:-5] ##ever data will be the training set except the last 5 data which are used as test set

##Training the dataset using the svm classifier
model.fit(x,y)

##Data chosed for prediction using the classifier
imageNumber = -4
predictedImage = numberImages.data[imageNumber]

##Priting out the predicted image
print "The predicted image is: ", model.predict(predictedImage)

##Plotting the predicted image
plt.matshow(numberImages.images[imageNumber],cmap=cm.Greys_r) 
plt.show()
