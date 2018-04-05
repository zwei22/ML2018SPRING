import numpy as np
import math
import sys

def sigmoid(x):
    return 1/(1+math.exp(-x))

def f(x):
    return sigmoid((np.dot(x, w)).sum())

def test():
    global nRows
    output = np.zeros(nRows)

    for idx in range(nRows):
        output[idx] = f(test_x[idx])
        output = np.around(output)

    return output
    
def feature_sc():
    global test_x

    for col in range(test_x.shape[1]):
        max = np.amax(test_x[:, col])
        if max > 1:
            test_x[:, col] /= max
        #print(col, '\t', max)
  
def missing_value():
    global test_x

    test_x[:, 106] += test_x[:, 117]
    test_x = np.delete(test_x, 117, 1)

    test_x[:, 9] += test_x[:, 8]
    test_x = np.delete(test_x, 8, 1)
      
if __name__ == '__main__':

    test_x = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
    test_x = np.hstack((np.ones([test_x.shape[0], 1]), test_x))
     
    missing_value()

    test_x = np.hstack((test_x[:, :11], test_x[:, 27:]))

    nRows = test_x.shape[0] 

    feature_sc() 

    w = np.load('w.npy')
    print('weight:')
    for idx in range(w.shape[0]):
        print(idx, '\t', w[idx])

    output = test().astype(int)
    with open(sys.argv[2], "w") as of:
        of.write('id,label\n')
        for idx in range(nRows):
            of.write(str(idx+1))
            of.write(',')
            of.write(str(output[idx]))
            of.write('\n')
