from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np

def feature_sc():
    global train_x

    norm = np.zeros([5, 2])
    c = 0
    for idx in [0, 9, 61, 62, 63]:
        norm[c, 0] = np.mean(train_x[:, idx])
        norm[c, 1] = np.std(train_x[:, idx])
        train_x[:, idx] -= norm[c, 0] 
        train_x[:, idx] /= norm[c, 1]
        c +=1

    np.save('norm.npy', norm)

def missing_value():
    global train_x

    train_x[:, 105] += train_x[:, 116]
    train_x = np.delete(train_x, 116, 1)
    train_x[:, 8] += train_x[:, 7]
    train_x = np.delete(train_x, 7, 1)

def main():
    global train_x
    global train_y
    train_x = np.genfromtxt('train_X', delimiter=',', skip_header=1)
    train_y = np.genfromtxt('train_Y', delimiter=',')

    missing_value()
    train_x = np.hstack((train_x[:, :10], train_x[:, 26:]))
    feature_sc()
    
    for idx in [0, 9, 61, 62, 63]:
        for pwr in range(2, 7):
            train_x = np.hstack((train_x, (train_x[:, idx]**pwr).reshape(train_x.shape[0], 1)))
    
    classifier = LogisticRegression(penalty='l1')
    classifier.fit(train_x, train_y)
    print(classifier.score(train_x, train_y))
    joblib.dump(classifier, 'model.pkl')
    
if __name__ == '__main__':
    main()
