# Various utils for different bits of code

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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

def plotModel(X, y, model, size_of_points=100):
    X = np.array(X)
    y = np.array(y)
    plot_step = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pyplot.contourf(xx, yy, Z, colors=['red', 'blue'], alpha=0.2, levels=range(-1,2))
    pyplot.contour(xx, yy, Z,colors = 'k',linewidths = 1)
    plotPoints(X, y)
    pyplot.show()


