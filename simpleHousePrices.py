# Unfinished code for line of regression 
# Haven't figured out matplotlib yet

import matplotlib as plt
import numpy as np
import random
import utils  

def squTrick(basePrice, roomPrice, numRooms, price, learningRate): 
    predictPrice = roomPrice*numRooms + basePrice  # y = mx + c 
    basePrice += (price-predictPrice) + learningRate
    roomPrice += (price-predictPrice)*numRooms*learningRate 
    return roomPrice, basePrice 

def linearRegression(features, labels, learningRate=0.01, iterations=1000):
    basePrice = random.random()
    roomPrice = random.random()
    for iteration in range(iterations): 

        if True: 
            utils.draw_line(roomPrice, basePrice, starting=0, ending=0)

        index = random.randint(1, len(features)-1)
        numRooms = numberOfRooms[index]
        price = housePrices[index]
        squTrick(basePrice, roomPrice, numRooms, price, learningRate) 

    utils.draw_line(roomPrice, basePrice, "black", starting=0, ending=8)
    utils.plot_points(features, labels)
    print(f"Price per room {roomPrice}")
    print(f"Base price {basePrice}")
    return roomPrice, basePrice



plt.ylim(0,500)

numberOfRooms = np.array([1,2,3,5,6,7])
housePrices = np.array([155,197,244,356,407,448])
linearRegression(numberOfRooms, housePrices)
