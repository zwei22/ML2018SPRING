from sklearn.externals import joblib
import numpy as np
import sys

def feature_sc():
    global test_x
    norm = np.load('norm.npy')

    c = 0
    for idx in [0, 9, 61, 62, 63]:
        test_x[:, idx] -= norm[c, 0] 
        test_x[:, idx] /= norm[c, 1]
        c +=1
  
def missing_value():
    global test_x

    test_x[:, 105] += test_x[:, 116]
    test_x = np.delete(test_x, 116, 1)

    test_x[:, 8] += test_x[:, 7]
    test_x = np.delete(test_x, 7, 1)
      

def main():
    global test_x
    test_x = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
     
    missing_value()
    test_x = np.hstack((test_x[:, :10], test_x[:, 26:]))
    feature_sc() 

    for idx in [0, 9, 61, 62, 63]:
        for pwr in range(2, 7):
            test_x = np.hstack((test_x, (test_x[:, idx]**pwr).reshape(test_x.shape[0], 1)))

    model = joblib.load('model.pkl')
    output = model.predict(test_x)
    output = output.astype(int)

    with open(sys.argv[2], "w") as of:
        of.write('id,label\n')
        for idx in range(test_x.shape[0]):
            of.write(str(idx+1))
            of.write(',')
            of.write(str(output[idx]))
            of.write('\n')

if __name__ == '__main__':
    main()
