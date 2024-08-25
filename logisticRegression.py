# This is my manual "logistic regression" algorithm

import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import numpy as np
import random
import utils

features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[3,2],[2,3]])
labels = np.array([0,0,0,0,1,1,1,1]) 

#utils.plotPoints(features, labels)
#plt.show()

def sigmoid(x): 
    return np.exp(x)/(np.exp(x)+1) # Same as 1/(1+np.exp(-x)) but this is better for small floating point numbers

def score(weights, bias, features): 
    return np.dot(weights, features) + bias 

def prediction(weights, bias, features): 
    return sigmoid(score(weights, bias, features))

def logLoss(weights, bias, features, label): 
    pred = prediction(weights, bias, features)
    return -label*np.log(pred) - (1-label)*np.log(1-pred)

def totalLogLoss(weights, bias, features, labels): 
    total = 0 
    for i in range(len(features)): 
        total += logLoss(weights, bias, features[i], labels[i])

    return total 

def logisticTrick(weights, bias, features, label, learningRate = 0.01): 
    pred = prediction(weights, bias, features)
    for i in range(len(features)): 
        weights[i] += (label-pred)*features[i]*learningRate 
    bias += (label-pred)*learningRate 

    return weights, bias 

def logisticRegressionAlg(features, labels, epochs = 1000):
    utils.plotPoints(features, labels)
    weights = [1.0 for i in range(len(features[0]))]
    bias = 0
    #errors = []
    for i in range(epochs): 
        utils.drawLine(weights[0], weights[1], bias, color='grey', linewidth=0.1, linestyle='dotted')
        j = random.randint(1, len(features)-1)
        weights, bias = logisticTrick(weights, bias, features[j], labels[j])

    utils.drawLine(weights[0], weights[1], bias)
    plt.show()
    return weights, bias

print(logisticRegressionAlg(features, labels))

# This is logistic regression using turicreate 

import turicreate as tc 
import numpy as np
import utils
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[3,2],[2,3]])
labels = np.array([0,0,0,0,1,1,1,1]) 

data = tc.SFrame({'x1' : features[:,0], 'x2': features[:,1], 'y': labels})
#print(data)

classifier = tc.logistic_classifier.create(data, 
                                           features = ['x1','x2'],
                                           target = 'y',
                                           validation_set=None)

bias, w1, w2 = classifier.coefficients['value']

utils.plotPoints(features, labels)
utils.drawLine(w1, w2, bias)
plt.show()

# Classifying IMDB movie reviews 

import turicreate as tc

movies = tc.SFrame('./IMDB_Dataset.csv')  # Find on Luis Serrano git 

movies['Words'] = tc.text_analytics.count_words(movies['review'])  # Places all words in dict 'Words' with letter count 
model = tc.logistic_classifier.create(movies, features=['words'], target='sentiment')  # Makes logistic classifier on words about sentiment (P/N) 

weights = model.coefficients 
print(weights) 

weights.sort('value', ascending=False)  # From highest to lowest 
weights[weights['index']=='wonderful']  # Gives value of wonderful 


















