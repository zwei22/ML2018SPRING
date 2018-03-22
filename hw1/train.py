import numpy as np
# PM2.5 is at col 9

def gradient_decent():
    global train_data
    global w
    global b
    global var_w
    global var_b
    global eta
    global lam

    loss = 0
    gradient_w = np.zeros([9, train_data.shape[2]])
    gradient_b = np.zeros([1])
    for month in train_data:
        for idx in range(month.shape[0]-9):
            delta = month[idx+9, 0] - (month[idx:idx+9]*w).sum() - b
            loss += delta**2
            #loss += delta**2 + lam*(w**2).sum()
            gradient_w -= delta*month[idx:idx+9]
            #gradient_w += -delta*month[idx:idx+9] + lam*w
            gradient_b -= delta


    var_w += gradient_w**2 
    var_b += gradient_b**2 
    w -= gradient_w*eta/var_w**0.5
    b -= gradient_b*eta/var_b**0.5

    return loss

def preprocessing(i):
    global all_data
    for m in range(all_data.shape[0]):
        for t in range(all_data.shape[1]):
            if all_data[m, t, i] > 300 or all_data[m, t, i] < 0:
                #print(m, ',', t, ',', test_data[m, t, 9])
                all_data[m, t, i] = 0
    for m in range(all_data.shape[0]):
        for t in range(all_data.shape[1]):
            if all_data[m, t, i] == 0:
                count = 0
                p = all_data[m, t-1, i]
                n = all_data[m, t+count, i]
                while n == 0:
                    count += 1
                    n = all_data[m, t+count, i]
                for idx in range(count):
                    all_data[m, t+idx, i] = (p*(count-idx)+n*(idx+1))/(count+1)
                    #print(m, ',', t, ',', count, ',', all_data[m, t+idx, 9])

if __name__ == '__main__':
    all_data = np.genfromtxt('train1.csv', delimiter=',').reshape(240, 18, 24).transpose(0, 2, 1).reshape(12, 480, 18)

    preprocessing(8)
    preprocessing(9)

    pm25 = all_data[..., 9]
    pm10 = all_data[..., 8]
    pm25sq = all_data[..., 9]**2
    pm10sq = all_data[..., 8]**2

    #train_data = np.stack((pm25, pm10, pm25sq, pm10sq), axis = -1)
    train_data = np.stack((pm25, pm10), axis = -1)
    #train_data = pm25

    w = np.random.rand(9, train_data.shape[2])
    b = np.random.rand(1)
    eta = 100
    lam = 10
    var_w = np.zeros([9, train_data.shape[2]]) 
    var_b = np.zeros([1]) 

    for i in range(15000):
        loss = gradient_decent()
        if i%10 == 0:
            print(i,'\t\t',loss)
   
    np.save('w', w)
    np.save('b', b)
