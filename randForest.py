# Learning to use the random forest classifier

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import utils 

from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0) 

emails = np.array([
    [7,8,1],
    [3,2,0],
    [8,4,1],
    [2,6,0],
    [6,5,1],
    [9,6,1],
    [8,5,0],
    [7,1,0],
    [1,9,1],
    [4,7,0],
    [1,3,0],
    [3,10,1],
    [2,2,1],
    [9,3,0],
    [5,3,0],
    [10,1,0],
    [5,9,1],
    [10,8,1],
])

spam_dataset = pd.DataFrame(data=emails, columns=["Lottery", "Sale", "Spam"])
features = spam_dataset[["Lottery","Sale"]]
labels = spam_dataset["Spam"]

#utils.plotPoints(features, labels)

# Overfitting Tree
'''decision_tree_classifier = DecisionTreeClassifier(random_state=0) 
decision_tree_classifier.fit(features, labels)
decision_tree_classifier.score(features, labels) 

utils.plotModel(features, labels, decision_tree_classifier)'''

# Manually Building Ensemble (Strong learner using multiple weak learners)
'''first_batch = spam_dataset.loc[[0,1,2,3,4,5]] 
features1 = first_batch[["Lottery", "Sale"]]
labels1 = first_batch["Spam"] 

second_batch = spam_dataset.loc[[6,7,8,9,10,11]] 
features2 = second_batch[["Lottery", "Sale"]]
labels2 = second_batch["Spam"] 

third_batch = spam_dataset.loc[[12,13,14,15,16,17]] 
features3 = third_batch[["Lottery", "Sale"]]
labels3 = third_batch["Spam"]  

dt1 = DecisionTreeClassifier(random_state=0, max_depth=1)
dt1.fit(features1, labels1)
utils.plotModel(features1, labels1, dt1)

dt2 = DecisionTreeClassifier(random_state=0, max_depth=1)
dt2.fit(features2, labels2)
utils.plotModel(features2, labels2, dt2)

dt3 = DecisionTreeClassifier(random_state=0, max_depth=1)
dt3.fit(features3, labels3)
utils.plotModel(features3, labels3, dt3)'''

# Strong Learner using randomForestClassifier
random_forest_classifier = RandomForestClassifier(random_state=0, n_estimators=5, max_depth=1) 
random_forest_classifier.fit(features, labels)
random_forest_classifier.score(features, labels)
utils.plotModel(features, labels, random_forest_classifier)

random_forest_classifier2 = RandomForestClassifier(random_state=0, n_estimators=5, max_depth=1) 
random_forest_classifier2.fit(features, labels)
random_forest_classifier2.score(features, labels)
utils.plotModel(features, labels, random_forest_classifier2)


