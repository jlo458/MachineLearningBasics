# Introductions to decision trees 
# Uses plotModel and plotPoints functions from utils to create diagragm 

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import utils

dataset = pd.DataFrame({ 
    'x_0':[7,3,2,1,2,4,1,8,6,7,8,9],
    'x_1':[1,2,3,5,6,7,9,10,5,8,4,6],
    'y':[0,0,0,0,0,0,1,1,1,1,1,1]
})

features = dataset[['x_0','x_1']]
labels = dataset['y']

#utils.plotPoints(features, labels)

decisionTree = DecisionTreeClassifier()
decisionTree.fit(features, labels)
decisionTree.score(features, labels)

utils.plotModel(features, labels, decisionTree)
