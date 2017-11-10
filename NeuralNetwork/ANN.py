import pandas as pd 
import numpy as np 
import argparse
import random
import logging

learning_rate = 0.01
def log(): 
	logging.basicConfig()
	logger = logging.getLogger('ANN')
	logger.setLevel(logging.DEBUG)
	return logger

# read dataset into memory given the full path to the dataset
def read(path): 
	logger.debug("Reading dataset %s" % path)
	data = pd.read_csv(path, header=0).as_matrix()
	logger.debug("Finished reaing dataset %s" % path)
	return data
# randomly split data as training set and test set given the full dataset and percentage
def split(data, percent): 
	logger.debug("Start to split dataset")
	nrow = len(data)
	ratio = float(percent) / 100
	train = []
	test = []
	for i in range(nrow): 
		if(random.random()<ratio): 
			train.append(data[i])
		else: test.append(data[i])
	logger.debug("Finished splitting dataset")
	logger.debug("Size of training set: %d, Size of test set: %d" % (len(train), len(test)))
	return np.matrix(train), np.matrix(test)

# initialization of weights of between two layers 
# given # of nodes in previous layer and # of nodes in next layer
def init(prev, next): 
	return np.random.rand(prev, next)

# initialization of weights for the full ANN
def initWeights(ncol, hidden, neuron): 
	logger.debug("Initialization of weights for ANN")
	# num of nodes in each layer(hidden and output)
	nodes = []
	for i in range(hidden): 
		nodes.append(neuron) 
	nodes.append(1) # the last output layer
	# initialize weights given number of nodes in the next level
	weights = []
	curNum = ncol
	for i in range(len(nodes)): 
		nextNum = nodes[i]
		weights.append(init(curNum, nextNum))
		curNum = nextNum
	logger.debug("Finished Initializtion of weights for ANN")
	return weights

# sigmoid function as the output function given the net x
def sigmoid(x): 
	return np.array(1 / (1 + np.exp(x)))

# foward process in back-propogation algorithm
def forward(sample, weights): 
	data = np.array(sample)
	output = []
	output.append(data)
	for i in range(len(weights)):
		data = np.dot(data, weights[i])
		data = sigmoid(data)
		output.append(np.array(data))
	return output
# backward process in back-propagation algorithm
def backward(sample_output, forward_outputs, weights, learning_rate): 
	deltas = []
	n = len(forward_outputs)
	for i in range(n - 1, -1, -1):
		#print("deltas: ", deltas)
		o = forward_outputs[i]
		if(i == n -  1): 
			delta = np.multiply(o, 1 - o)
			delta = np.multiply(delta, sample_output - o)
			deltas.insert(0, delta)		
		else:
			delta = deltas[0]
			delta_w = np.multiply(learning_rate, np.multiply(o.transpose(), delta))
			new = np.multiply(o, 1 - o)
			new = np.multiply(new, np.dot(delta, weights[i].transpose()))
			deltas.insert(0, new)
			weights[i] = weights[i] - delta_w
	return
# full implementation of backpropogation algorithm
def backpropogation(trainX, trainY, hidden, neuron, learning_rate, iteration): 
	logger.debug("Started to training data using Back-propogation algorithm")
	weights = initWeights(trainX.shape[1], hidden, neuron)
	nrow = trainX.shape[0]
	for i in range(iteration): 
		for j in range(nrow): 
			output = forward(trainX[i], weights)
			backward(trainY[i], output, weights, learning_rate)
	logger.debug("Finished training data using Back-propogation algorithm")
	return weights; 

if __name__ == '__main__': 
	# argument parser
	parser = argparse.ArgumentParser() 
	parser.add_argument('path', help = 'the full path to input dataset')
	parser.add_argument('percent', help = 'training percent of the input dataset')
	parser.add_argument('iteration', help = 'maximum number of iterations of ANN')
	parser.add_argument('hidden', help = '# of hidden layers')
	parser.add_argument('neuron', help = '# of neuron in each hidden layer')

	args = parser.parse_args()
	inPath = args.path
	percent = int(args.percent)
	iteration = int(args.iteration)
	hidden = int(args.hidden)
	neuron = int(args.neuron)

	logger = log()
	train, test = split(read(inPath), percent)
	trainX = train[:,:-1]
	trainY = train[:,-1]
	testX = test[:,:-1]
	testY = test[:,-1]

	#initiate weights for ANN
	'''
	weights = initWeights(trainX.shape[1], hidden, neuron)
	print(weights)
	output = forward(trainX[0], weights)
	backward(trainY[0], output, weights, learning_rate)
	'''
	weights = backpropogation(trainX, trainY, hidden, neuron, learning_rate, iteration)
	print(weights)




