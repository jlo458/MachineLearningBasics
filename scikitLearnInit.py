# Utils slightly tweaked 
# Uses linear.csv file

from sklearn.svm import SVC 
import numpy as np
import pandas as pd 
import utils 

#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt 

#plt.use('TkAgg')

'''linear_data = pd.read_csv("linear.csv") 
features = np.array(linear_data[['x_1', 'x_2']])
labels = np.array(linear_data['y']) '''

poly_data = pd.read_csv("two_circles.csv")
features = np.array(poly_data[['x_1','x_2']])
labels = np.array(poly_data['y'])

#utils.plotPoints(features, labels)

'''svm_linear = SVC(kernel='linear')
svm_linear.fit(features, labels)
print("Accuracy:", svm_linear.score(features, labels))'''

'''# C = 0.01 - C value determines what to prioritise (accuracy or avg distance)
svm_c_001 = SVC(kernel='linear', C=0.01)
svm_c_001.fit(features, labels)
print("C=0.01")
print("Accuracy:", svm_c_001.score(features, labels))
utils.plotModel(features, labels, svm_c_001)

# C = 100 
svm_c_100 = SVC(kernel='linear', C=100)
svm_c_100.fit(features, labels)
print("C=100")
print("Accuracy:", svm_c_100.score(features, labels))
utils.plotModel(features, labels, svm_c_100)'''

'''# Polynomial - degree determines degree of polynomial created
svm_p = SVC(kernel='poly', degree=4)
svm_p.fit(features, labels)
print("poly")
print("Accuracy:", svm_p.score(features, labels))
utils.plotModel(features, labels, svm_p)'''


# RBF Kernel - change gamma for over/underfitting
svm_gamma = SVC(kernel='rbf', gamma=4)
svm_gamma.fit(features, labels)
print("rbf")
print("Accuracy:", svm_gamma.score(features, labels))
utils.plotModel(features, labels, svm_gamma)
