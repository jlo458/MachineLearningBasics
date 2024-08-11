import numpy as np
import matplotlib
from matplotlib import pyplot

def plotPoints(features, labels): 
    X = np.array(features)
    y = np.array(labels)
    ham = X[np.argwhere(y==1)]
    spam = X[np.argwhere(y==0)]
    pyplot.scatter([s[0][0] for s in spam], 
                   [s[0][1] for s in spam],
                   s = 100, 
                   color = 'red',
                   edgecolors = 'k',
                   marker = 'o'
                   )
    
    pyplot.scatter([s[0][0] for s in ham], 
                   [s[0][1] for s in ham],
                   s = 100, 
                   color = 'cyan',
                   edgecolors = 'k',
                   marker = '^')
    
    pyplot.xlabel("aack")
    pyplot.ylabel("beep")
    pyplot.legend(['Sad', 'Happy'])

def drawLine(a,b,c, starting=0, ending=3, **kwargs): 
    x = np.linspace(starting, ending, 1000)
    pyplot.plot(x, (-a*x-c)/b, **kwargs)
