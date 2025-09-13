# Using gradient boosting and XGBoost for regression

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import utils 

from sklearn.ensemble import GradientBoostingRegressor 

features = np.array([[10],[20],[30],[40],[50],[60],[70],[80]])
labels = np.array([7,5,7,1,2,1,5,4])

'''plt.scatter(features, labels)
plt.xlabel("Age")
plt.ylabel("No. Hours")
plt.show()'''

gradBoostRegressor = GradientBoostingRegressor(max_depth=2, n_estimators=4, learning_rate=0.6)
gradBoostRegressor.fit(features, labels)
predictions = gradBoostRegressor.predict(features)  

for i in range(len(predictions)): 
    print(f"Age {(i+1)*10} | Prediction: {predictions[i]}")

utils.plot_regressor(features, labels, gradBoostRegressor)
