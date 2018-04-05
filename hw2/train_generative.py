import numpy as np
import math


def calc():
    global train_x, train_y

    nCols = train_x.shape[1]
    nRows = train_x.shape[0]

    c = np.zeros(2) 
    mu = np.zeros([2, nCols])

    for idx in range(nRows):
        if train_y[idx] == 0:
            mu[0] += train_x[idx]
            c[0] +=1
        else:
            mu[1] += train_x[idx]
            c[1] +=1
    mu[0] /= c[0]
    mu[1] /= c[1]

    sigma = np.zeros([2, nCols, nCols])

    for idx in range(nRows):
        if train_y[idx] ==0:
            sigma[0] += np.dot(np.transpose([train_x[idx] - mu[0]]), [train_x[idx] - mu[0]])
        else:
            sigma[1] += np.dot(np.transpose([train_x[idx] - mu[1]]), [train_x[idx] - mu[1]])

    sigma[0] /= c[0]
    sigma[1] /= c[1]

    sigma = float(c[0]) / nRows * sigma[0] + float(c[1]) / nRows * sigma[1]

    np.save('sigma.npy', sigma)
    np.save('mu.npy', mu)
    np.save('c.npy', c)

def main():
    global train_x, train_y

    train_x = np.genfromtxt('train_X', delimiter=',', skip_header=1)
    train_y = np.genfromtxt('train_Y', delimiter=',')

    calc()

if __name__ == '__main__':
    main()
