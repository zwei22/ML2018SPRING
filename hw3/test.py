import numpy as np
import io
from keras.models import load_model
import sys

def predict(x_test):
    output = np.zeros(x_test.shape[0]) 
    x_test /= 255
    x_test =  x_test.reshape(-1, 48, 48, 1);
    print('loading model')
    model = load_model('model.h5')
    result = model.predict(x_test)
    for i in range(x_test.shape[0]):
        output[i] = np.argmax(result[i])

    return output.astype('int16')

def read(filename):
    data = io.StringIO(open(filename).read().replace(',',' '))
    x_test = np.genfromtxt(data, delimiter=' ',  skip_header=1)[:, 1:]
    #np.save('x_test.npy', x_test)
    return x_test 

def read_from_npy():
    x_test = np.load('x_test.npy')
    return x_test

def write(filename, output):
    with open(filename, "w") as of:
        of.write('id,label\n')
        for i in range(output.shape[0]):
            of.write(str(i))
            of.write(',')
            of.write(str(output[i]))
            of.write('\n')

def main():
    x_test = read(sys.argv[1])
    #x_test = read_from_npy()
    result = predict(x_test)
    print(result)
    write(filename=sys.argv[2], output=result)

if __name__=="__main__":
    main()
