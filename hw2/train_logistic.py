import numpy as np
import math


def sigmoid(x):
    return 1/(1+math.exp(-x))

def f(x):
    return sigmoid((np.dot(x, w)).sum())

def crossentropy(x, y):
    return -(y*math.log(x) + (1-y)*math.log(1-x))

def gradient_decent():
    global train_x
    global train_y
    global nCols
    global nRows
    global w
    global var
    global eta

    loss = 0
    gradient = np.zeros(nCols)

    for idx in range(nRows):
        x = train_x[idx]
        y = train_y[idx]
        delta = y-f(x)
        gradient -= delta*train_x[idx]

    var += gradient**2
    w -= gradient*eta/var**0.5

def loss():
    global train_x
    global train_y
    global nRows

    loss = 0
    for idx in range(nRows):
        x = train_x[idx]
        y = train_y[idx]
        loss += crossentropy(f(x), y) 

    return loss

def feature_sc():
    global train_x
    train_x = train_x

    for col in range(train_x.shape[1]):
        max = np.amax(train_x[:, col])
        if max > 1:
            train_x[:, col] /= max
        #print(col, '\t', max)

def missing_value():
    global train_x
    train_x = train_x

    train_x[:, 106] += train_x[:, 117]
    train_x = np.delete(train_x, 117, 1)
    train_x[:, 9] += train_x[:, 8]
    train_x = np.delete(train_x, 8, 1)



if __name__ == '__main__':
    train_x = np.genfromtxt('train_X', delimiter=',', skip_header=1)
    train_y = np.genfromtxt('train_Y', delimiter=',')
    train_x = np.hstack((np.ones([train_x.shape[0], 1]), train_x))

    missing_value()

    train_x = np.hstack((train_x[:, :11], train_x[:, 27:]))

    nCols = train_x.shape[1] 
    nRows = train_x.shape[0] 

    w = np.random.rand(nCols)
    eta = 1
    var = np.zeros(nCols)

    feature_sc() 

    for i in range(2000):
        gradient_decent()
        if i%10 == 0:
            print(i,'\t\t',loss())
    
    np.save('w', w)
    print('weight:')
    print(w)
