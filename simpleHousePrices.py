# First ml program 
# Calculates line of regression using "square trick" for correlation between house price & no. rooms 
# This has now been updated to stop when RMSE function stops changing (as much) 
# A simpler version without the RMSE can be found on my "python_practice" repo

import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque

def RMSE(labels, predictions): 
  n = len(labels) 
  differences = np.subtract(labels, predictions)
  return (1/n * (np.dot(differences, differences)))

def squTrick(basePrice, roomPrice, numRooms, price, learningRate): 
    predictPrice = roomPrice*numRooms + basePrice  # y = mx + c 
    basePrice += (price-predictPrice)*learningRate
    roomPrice += (price-predictPrice)*numRooms*learningRate 
    return roomPrice, basePrice 

def linearRegression(features, labels, errors=deque(), learningRate=0.01):
    basePrice = random.random()
    roomPrice = random.random()

    finish = False
    count = 0

    while not finish and count < 5000:
        predictVals = np.empty(len(features), dtype=float)
        for ind in range(len(features)): 
            predictVal = roomPrice*(features[ind]) + basePrice
            predictVals[ind] = predictVal 

        squError = RMSE(labels, predictVals)
        errors.append(squError)

        if len(errors) > 5: 
            avg = 0
            for val in errors: 
                avg += val

            avg = avg / 5 
            if avg < 200: 
                finish = True 

            errors.popleft()
        
        index = random.randint(1, len(features)-1)
        numRooms = features[index]
        price = labels[index] 
        roomPrice, basePrice = squTrick(basePrice, roomPrice, numRooms, price, learningRate) 
        count += 1
        
    print(errors, count)
    return roomPrice, basePrice

numberOfRooms = np.array([1,2,3,5,6,7])
housePrices = np.array([155,197,244,356,407,448])
errors = deque()
m, c = linearRegression(numberOfRooms, housePrices)

labelVals = np.empty(len(numberOfRooms), dtype=float)
for ind in range(len(numberOfRooms)): 
    labelVal = m*(numberOfRooms[ind]) + c 
    labelVals[ind] = labelVal
    

plt.plot(numberOfRooms, labelVals, "r")
plt.plot(numberOfRooms, housePrices, marker="*")
plt.show() 

inp = float(input("Enter number of rooms: "))
predictedPrice = m*inp + c
print(f"Predicted house price is £{predictedPrice}")
