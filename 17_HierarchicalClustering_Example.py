####This source code gives a simple demosntration of Hierarchical Clustering

##Import statements
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt

##Creating the dataset
x = np.array([[1,1],[1.1,1.1],[3,3],[4,4],[3,3.5],[3.5,4]])

##Creating a scatter plot out of the dataset
plt.scatter(x[:,0], x[:,1], s= 50)

##Plot show command
plt.show()

##Performing Hierarchical Clustering on the dataset and extracting the linkage matrix
linkage_matrix = linkage(x, 'single')

##Printing out the linkage_matrix
print "Shape of linkage matric: ", linkage_matrix.shape
print "\n"
print "Linkage Matrix: ", "\n", linkage_matrix
print "\n"

##Creating the dendrogram of the clustering analysis
dendrogram = dendrogram(linkage_matrix, truncate_mode = 'none')

##Giving the title of the plot
plt.title("Hierarchical Clustering")

##Plot show command
plt.show()