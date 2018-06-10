import numpy as np
import sys
from keras.models import load_model

def predict(x_test, model_path, norm=False):
    result = np.zeros(len(x_test[0])) 
    model = load_model(model_path)
    result = model.predict(x_test).squeeze()
    if norm:
        result *= 2
        result += 3
    print(result.shape)
    return result 

def read_data(path, x_test):
    data = np.genfromtxt(path, delimiter=',', dtype='int32', skip_header=1)
    x_test['user'] = data[:, 1] 
    x_test['movie'] = data[:, 2] 
    return x_test

def read_movie(path, x_test):
    movie = np.load(path).astype('int32')
    x_test['year'] = movie[x_test['movie'], 0]
    x_test['cat'] = movie[x_test['movie'], 1:]
    return x_test 

def read_user(path, x_test):
    user = np.load(path).astype('int32')
    x_test['gender'] = user[x_test['user'], 0]
    x_test['age'] = user[x_test['user'], 1]
    x_test['occu'] = user[x_test['user'], 2]
    return x_test

def write(filename, output):
    with open(filename, "w") as of:
        of.write('TestDataID,Rating\n')
        for i in range(output.shape[0]):
            of.write('{0},{1}\n'.format(i+1, output[i]))

def main():
    x_test = dict()
    x_test = read_data(sys.argv[1], x_test)
    use = ['user', 'movie']
    x_test = [x_test[key] for key in use]
    result1 = predict(x_test, model_path=sys.argv[3], norm=True)
    result2 = predict(x_test, model_path=sys.argv[4], norm=True)
    result = (result1*0.5+result2*0.5).clip(1, 5)
    write(sys.argv[2], result)

if __name__=='__main__':
    main()
