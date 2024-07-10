# First ml program 
# Calculates line of regression using "square trick" for correlation between house price & no. rooms

import matplotlib.pyplot as plt
import numpy as np
import random

def squTrick(basePrice, roomPrice, numRooms, price, learningRate): 
    predictPrice = roomPrice*numRooms + basePrice    # y = mx + c 
    basePrice += (price-predictPrice)*learningRate    # Alters base price (c)
    roomPrice += (price-predictPrice)*numRooms*learningRate    # Alters gradient (m)
    return roomPrice, basePrice 

def linearRegression(features, labels, learningRate=0.01, iterations=5000):
    basePrice = random.random()
    roomPrice = random.random()
    for _ in range(iterations): 
        index = random.randint(1, len(features)-1)
        numRooms = features[index]
        price = labels[index]         
        roomPrice, basePrice = squTrick(basePrice, roomPrice, numRooms, price, learningRate) 
        print(roomPrice, basePrice)

  
    print(f"Price per room {roomPrice}")
    print(f"Base price {basePrice}")
    return roomPrice, basePrice


numberOfRooms = np.array([1,2,3,5,6,7])
housePrices = np.array([155,197,244,356,407,448])    # Starting values
m, c = linearRegression(numberOfRooms, housePrices)

labelVals = np.empty(len(numberOfRooms), dtype=float)
for ind in range(len(numberOfRooms)): 
    labelVal = m*(numberOfRooms[ind]) + c 
    labelVals[ind] = labelVal
    
plt.plot(numberOfRooms, labelVals, "r")
plt.plot(numberOfRooms, housePrices, marker="*")
plt.show()

# Now we can predict house prices based on the number of rooms
inp = float(input("Enter number of rooms: "))
predictedPrice = m*inp + c
print(f"Predicted house price is £{predictedPrice}")
