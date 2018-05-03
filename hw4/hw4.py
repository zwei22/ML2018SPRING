from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import sys

def write(filename, output):
    with open(filename, "w") as of:
        of.write('ID,Ans\n')
        for i in range(output.shape[0]):
            of.write('{0},{1}\n'.format(i, output[i]))
            
def read(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype='int16', skip_header=1)[:, 1:]
    return data
    
def main():
    data = np.load(sys.argv[1])
    test = read(sys.argv[2])
    #test = np.load('./test.npy')
    result = np.zeros((test.shape[0])).astype('int16')

    pca = PCA(n_components=400, whiten=True, random_state=0)
    pca_result = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=2, random_state=100)
    kmeans_result = kmeans.fit(pca_result).labels_
    print(kmeans_result.sum())

    for i in range(test.shape[0]):
        result[i] = 1 if kmeans_result[test[i][0]]==kmeans_result[test[i][1]] else 0

    write(sys.argv[3], result)

if __name__=='__main__':
    main()
