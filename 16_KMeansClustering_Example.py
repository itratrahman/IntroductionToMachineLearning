####This sourcecode gives a simple demonstration of K Means Clustering

##Import statements
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

##Parameters for the random data set
noOfSamples = 1000
noOfClusters = 6
stdOfCluster = 0.5

##Creating clusters of random data set
x, y = make_blobs(n_samples = noOfSamples, centers = noOfClusters, random_state = 0, cluster_std = stdOfCluster)

##Plotting a scatter plot out of the data
plt.scatter(x[:,0],x[:,1],s = 50)

##Plot show command
plt.show()

##Variable to store the number of clusters
k = 6

##Instantiating a k means clustering classifier object
estimator = KMeans(k)

##Fitting the data with clustering classifier
estimator.fit(x)

##Generating the vector of clusters classification for the x data set
y_means = estimator.predict(x)

##Plotting the clusters with each cluster being color coded
plt.scatter(x[:,0], x[:,1], c = y_means, cmap = 'rainbow')

##Plot show command
plt.show()