####This sourcecode gives a simple demonstration of Neural Network solving XOR problem

##Import statements for neural network
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

##Setting the architecture of the neural network
neuralNetwork = buildNetwork(2, 3, 1)

##Setting the dimension of the supervised data set
ds = SupervisedDataSet(2, 1)

##Setting the supoervised data set
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))


##Instantiating the nerual netwrok backpropagation trainer 
#using the given architecture and data set  
trainer = BackpropTrainer(neuralNetwork, ds)

##Exectuing the backpropagation trainer on each iteration
#and printing the result of the backpropagation on 1000th iteration:
for i in range(1,10000):
	
	##Executing backpropagation
	trainer.train()
	
	if i % 1000 == 0:
			print(neuralNetwork.activate([0, 0]))
			print(neuralNetwork.activate([1, 0]))
			print(neuralNetwork.activate([0, 1]))
			print(neuralNetwork.activate([1, 1]))
