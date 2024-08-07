# Makes predictions based on given data using L1/L2 regularisation

import turicreate as tc
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

random.seed(0)

# Initial equation -x^2 + x + 15
coefs = [15, 1, -1]

def polynomial(coefs, x):
    n = len(coefs)
    return sum([coefs[i]*x**i for i in range(n)])


def drawPolynomial(coefs): 
    x = np.linspace(-5, 5, 1000)
    y = polynomial(coefs, x)
    plt.ylim(-20, 20)
    plt.plot(x, y, linestyle='-', color='black')
    plt.title('Plot of the Polynomial')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()


X = []
Y = []

for _ in range(40): 
    x = random.uniform(-5,5)
    y =  polynomial(coefs, x) + random.gauss(0, 2)
    X.append(x)
    Y.append(y)

plt.scatter(X, Y)

data = tc.SFrame({'x':X, 'y':Y})

for i in range(2,200): 
    string = 'x^'+str(i)
    data[string] = data['x'].apply(lambda x:x**i)

train, test = data.random_split(.8, seed=0)

def dispResults(model): 
    coefs = model.coefficients
    print("Training Error: ", model.evaluate(train)['rmse'])
    print("Test Error: ", model.evaluate(test)['rmse'])
    plt.scatter(train['x'], train['y'], marker='o')
    plt.scatter(test['x'], test['y'], marker='+')
    drawPolynomial(coefs['value'])
    plt.show()
    print("Polynomial Coefficients")
    print(coefs['name','value'])

# No Regularisation 
modelNoReg = tc.linear_regression.create(
    train, target='y', l1_penalty=0.0, l2_penalty=0.0, verbose=False, validation_set=None
)

modelL1Reg = tc.linear_regression.create(
    train, target='y', l1_penalty=0.1, l2_penalty=0.0, verbose=False, validation_set=None
)

modelL2Reg = tc.linear_regression.create(
    train, target='y', l1_penalty=0.0, l2_penalty=0.1, verbose=False, validation_set=None
)

# Shows Results of Model

# dispResults(modelL1Reg)

# Comparing Predictions

predictions = test['x','y']
predictions['No Reg'] = modelNoReg.predict(test)
predictions['L1 Reg'] = modelL1Reg.predict(test)
predictions['L2 Reg'] = modelL2Reg.predict(test)

print(predictions)


