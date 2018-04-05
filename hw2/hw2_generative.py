import numpy as np
import math
import sys

def sigmoid(x):
    return 1/(1+math.exp(-x))

def predict():
    global test_x
    global nRows

    sigma = np.load('sigma.npy')
    mu = np.load('mu.npy')
    c = np.load('c.npy')

    sigma_inv = np.linalg.pinv(sigma)
    w = np.dot((mu[1]-mu[0]), sigma_inv)
    x = test_x.T
    b = (-0.5)*np.dot(np.dot([mu[1]], sigma_inv), mu[1]) \
        +(0.5)*np.dot(np.dot([mu[0]], sigma_inv), mu[0]) \
        +np.log(float(c[1])/c[0])
    a = np.dot(w, x) + b

    output = np.zeros(nRows)
    for idx in range(nRows):
        output[idx] = sigmoid(a[idx])
    output = np.around(output)

    return output 

def main():
    global test_x
    global nRows
    test_x = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
    nRows = test_x.shape[0] 

    output = predict().astype(int)
    with open(sys.argv[2], "w") as of:
        of.write('id,label\n')
        for idx in range(nRows):
            of.write(str(idx+1))
            of.write(',')
            of.write(str(output[idx]))
            of.write('\n')

if __name__ == '__main__':
    main()
