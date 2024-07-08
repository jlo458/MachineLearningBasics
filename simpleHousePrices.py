# Unfinished code for line of regression 
# Haven't figured out matplotlib yet

import matplotlib.pyplot as plt
import numpy as np
import random

def squTrick(basePrice, roomPrice, numRooms, price, learningRate): 
    predictPrice = roomPrice*numRooms + basePrice  # y = mx + c 
    basePrice += (price-predictPrice) + learningRate
    roomPrice += (price-predictPrice)*numRooms*learningRate 
    return roomPrice, basePrice 

def linearRegression(features, labels, learningRate=0.01, iterations=1000):
    basePrice = random.random()
    roomPrice = random.random()
    for iteration in range(iterations): 

        labelVals = np.empty(len(numberOfRooms), dtype=int)
        #print(labelVals)
        for ind in range(len(numberOfRooms)): 
            labelVal = roomPrice*(numberOfRooms[ind]) + basePrice 
            labelVals[ind] = labelVal
            


        #print(f"{numberOfRooms} and then -- {labelVals}")
        #plt.plot(numberOfRooms, labelVals)
        #plt.show()

        index = random.randint(1, len(features)-1)
        numRooms = numberOfRooms[index]
        price = housePrices[index]
        squTrick(basePrice, roomPrice, numRooms, price, learningRate) 

  
    print(f"Price per room {roomPrice}")
    print(f"Base price {basePrice}")
    return roomPrice, basePrice



#plt.ylim(0,500)

numberOfRooms = np.array([1,2,3,5,6,7])
housePrices = np.array([155,197,244,356,407,448])
x, y = linearRegression(numberOfRooms, housePrices)

plt.plot(x,y,"r")
plt.show()
