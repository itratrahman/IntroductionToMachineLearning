####This sourcecode gives a demonstration of text clustering using naive bayes classifier

##Import Statements
import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

##Creating the four chosen target variables as a list
categories =  ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']

##Creating the training data by retrieving the data from corpus
trainingData = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
##Print the first 10 lines of a random email
# item = 0
# print "\n".join(trainingData.data[item].split("\n")[:10])
# print "Target is:", trainingData.target_names[trainingData.target[item]]

##To perform machine learning on texts we need to transform the text into numerical feature vectors
#and assign an integer id to each word. CountVectorizer method does that

##Instantiating a CountVectorizer object
countVectorizer = CountVectorizer()

##Executing the countVectorizer method on the training data set 
xTrainCounts = countVectorizer.fit_transform(trainingData.data)

##Finding the occurences of a specific word in the text for example 'software'
# print "Number of occurences: ", countVectorizer.vocabulary_.get(u'software')

##Tfidf should be executed to downscale the words that occur in many documents

##Instantiating a tfidf object
tfidfTransformer = TfidfTransformer()

##Executing the tfidf on the feature vectors that resulted 
##after execution of CountVectorizer method on the training dataset
xTrainTfidf = tfidfTransformer.fit_transform(xTrainCounts)

##Traning the data using multinomial Naive Bayes method 
#with the feature vectors and the training data targets(results)
model = MultinomialNB().fit(xTrainTfidf, trainingData.target)

##Test dataset containing two sentece elements
new = ['This has nothing to do with church or religion', 'Software engineering is geting hotter and hotter nowadays']

##Performing countVectorizer method on the test dataset
xNewCounts = countVectorizer.transform(new)

##Perfroming tfidf on the resulting feature vectors extracted from the test dataset
xNewTfidf = tfidfTransformer.transform(xNewCounts)

##Getting the prediction with the test dataset
predicted = model.predict(xNewTfidf)

##Iterating throught the test dataset and the predicted resulted
for doc, category in zip(new,predicted):
	print('%r --------> %s' % (doc, trainingData.target_names[category]))