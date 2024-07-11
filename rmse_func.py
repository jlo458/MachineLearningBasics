# RMSE Function 

def RMSE(labels, predictions): 
  n = len(labels) 
  differences = np.subtract(labels, predictions)
  return (1/n * (np.dot(differences, differences))
