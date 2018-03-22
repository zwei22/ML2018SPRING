import numpy as np
import sys

def test():
    global feature 
    global w
    global b
    output = np.zeros([feature.shape[0]])

    for idx in range(feature.shape[0]):
        output[idx] = (feature[idx, :]*w).sum() + b
    
    return output

def preprocessing(i):
    global test_data
    for m in range(test_data.shape[0]):
        for t in range(test_data.shape[1]):
            if test_data[m, t, i] > 300 or test_data[m, t, i] < 0:
                #print(m, ',', t, ',', test_data[m, t, idx])
                test_data[m, t, i] = 0
    for m in range(test_data.shape[0]):
        for t in range(test_data.shape[1]):
            if test_data[m, t, i] == 0:
                count = 0
                p = test_data[m, t-1, i]
                n = test_data[m, t+count, i]
                while n == 0:
                    count += 1
                    if t+count == test_data.shape[1]:
                        n = p
                    else:
                        n = test_data[m, t+count, i]
                for idx in range(count):
                    test_data[m, t+idx, i] = (p*(count-idx)+n*(idx+1))/(count+1)
                    #print(m, ',', t, ',', count, ',', test_data[m, t+idx, idx])

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('error')


    test_data = np.genfromtxt(sys.argv[1], delimiter=',')[:, 2:].reshape(260, 18, 9).transpose(0, 2, 1)

    preprocessing(8)
    preprocessing(9)

    w = np.load('w.npy')
    b = np.load('b.npy')

    pm25 = test_data[..., 9]
    pm10 = test_data[..., 8]
    pm25sq = test_data[..., 9]**2
    pm10sq = test_data[..., 8]**2
    #feature = pm25
    feature= np.stack((pm25, pm10), axis = -1)
    #feature = np.stack((pm25, pm10, pm25sq, pm10sq), axis = -1)
    
    print('weight:')
    print(w)
    print('bias:')
    print(b)
    output = test()
    with open(sys.argv[2], "w") as of:
        of.write('id,value\n')
        for idx in range(output.shape[0]):
            of.write('id_')
            of.write(str(idx))
            of.write(',')
            of.write(str(output[idx]))
            of.write('\n')
