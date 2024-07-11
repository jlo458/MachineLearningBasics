# RMSE Function 
# Can be used as a method to check when a program for linear regression should end

def RMSE(labels, predictions): 
  n = len(labels) 
  differences = np.subtract(labels, predictions)
  return (1/n * (np.dot(differences, differences))
