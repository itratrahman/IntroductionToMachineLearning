####This source code gives a demonstration of face recognition using neural netowork

##Import statements
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pylab import ion, ioff, figure, draw, contourf, clf, show, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from random import normalvariate
import numpy as np
from sklearn import datasets

# ---------------------------------------------------------------------
#
# Olivetti dataset -> 400 images ... 10 images / person 
#	There are ten different images of each of 40 distinct subjects. 
#	For some subjects, the images were taken at different times, varying the lighting, 
#	facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses)
#
#  So we have 40 target / 40 person !!!
#    64bit x 64bit images: number of features=64x64=4096
#

##Loading the training data set
olivettiData = datasets.fetch_olivetti_faces()

##Extracting the feature vector from the data set
dataFeatures = olivettiData.data

##Extracting the target variable
dataTargets = olivettiData.target

##Viewing an image sample from the data set, and printing out the corresponding target
# imageNumber = 0
# plt.matshow(olivettiData.images[imageNumber], cmap = cm.Greys_r)
# plt.show()
# print "Target Number: "
# print dataTargets[imageNumber]
# print "\n"

##Printing the shape of feature vector
# print "shape of the feature vector: "
# print dataFeatures.shape
# print "\n"

##Creating a matrix to store data set in format appropriate for classication purpose
dataSet = ClassificationDataSet(4096, 1 , nb_classes=40) 

##Storing the feature vector and the taget vraibles in the data set created above
for i in range(len(dataFeatures)):

	dataSet.addSample(np.ravel(dataFeatures[i]), dataTargets[i])


##Splitting the data set for cross validation purpose
testData, trainingData = dataSet.splitWithProportion(0.25)

trainingData._convertToOneOfMany()
testData._convertToOneOfMany()

##Setting the architecture of the neural network
neuralNetwork = buildNetwork(trainingData.indim, 64, trainingData.outdim, outclass=SoftmaxLayer) #activation function of the output layer is SoftmaxLayer

##Instantiating the nerual netwrok backpropagation trainer 
#using the given architecture and data set  
trainer = BackpropTrainer(neuralNetwork, dataset=trainingData, momentum=0.2, learningrate=0.01, verbose=True, weightdecay=0.02)

##Executing backpropagation for given number of time and print out the training set error each time
trainer.trainEpochs(50)

##Printing out the error on training set on the test data set
print 'Error (test dataset): ' , percentError(trainer.testOnClassData(dataset=testData), testData['class'])