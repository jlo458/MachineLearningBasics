# My perceptron algorithm

import matplotlib 
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import numpy as np
import random
import utils

features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[2,3],[3,2]])
labels = np.array([0,0,0,0,1,1,1,1])

def score(weights, bias, feature):
    return np.dot(feature, weights) + bias

def step(x):
    return 1 if x >= 0 else 0

def prediction(weights, bias, feature):
    return step(score(weights, bias, feature))

def error(weights, bias, feature, label):
    pred = prediction(weights, bias, feature)
    return 0 if pred == label else np.abs(score(weights, bias, feature))

def meanPerceptronError(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += error(weights, bias, features[i], labels[i])
    return total_error / len(features)

def perceptronTrick(w, b, f, l, learningRate = 0.01): 
    pred = prediction(w, b, f)
    for i in range(len(w)): 
        w[i] += (l - pred) * f[i] * learningRate
    b += (l - pred) * learningRate
    return w, b

random.seed(0)
def perceptronAlgorithm(features, labels, learning_rate = 0.01, epochs = 200):
    weights = np.ones(len(features[0]))
    bias = 0.0
    errors = []
    for epoch in range(epochs):
        # Uncomment the following line to draw only the final classifier
        utils.drawLine(weights[0], weights[1], bias, color='grey', linewidth=1.0, linestyle='dotted')
        error = meanPerceptronError(weights, bias, features, labels)
        errors.append(error)
        i = random.randint(0, len(features) - 1)
        weights, bias = perceptronTrick(weights, bias, features[i], labels[i], learning_rate)
    
    utils.drawLine(weights[0], weights[1], bias)
    utils.plotPoints(features, labels)
    plt.show()
    plt.scatter(range(epochs), errors)
    plt.show()
    return weights, bias

perceptronAlgorithm(features, labels)

# Or use turicreate 

import turicreate as tc
import numpy as np 

features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[2,3],[3,2]])
labels = np.array([0,0,0,0,1,1,1,1])

datadict = {'aack': features[:,0], 'beep': features[:,1], 'prediction': labels}
data = tc.SFrame(datadict)

perceptron = tc.logistic_classifier.create(data, target='prediction')
print(perceptron.coefficients)
