from skimage import io
import numpy as np
import sys, os
def eigenface():
    for i in range(4):
        M = U.T[i].reshape(600, 600, 3)
        M -= M.min()
        M /= M.max()
        M *= 225
        io.imsave('./faces/eg_{0}.jpg'.format(i), M.astype('uint8'))

def main():
    use = 4
    images = io.imread_collection(os.path.join(sys.argv[1], '*'))
    image_recons = io.imread(sys.argv[2])

    data = np.asarray(images).reshape(-1, 600*600*3).astype('float64')
    recons = np.asarray(image_recons).flatten().astype('float64')
    mean = np.mean(data, axis=0)
    #io.imsave("mean.jpg", mean.reshape(600, 600, 3).astype('uint8'))
    M = (data - mean).T
    U, s, vh = np.linalg.svd(M, full_matrices=False)
    V = vh.T
    S = np.diag(s)
    #print(s/s.sum())
    #np.save('U.npy', U)
    #np.save('S.npy', S)
    
    recons -= mean
    U_use = U[:, :use]
    Mhat = U_use.dot(U_use.T.dot(recons))
    Mhat += mean
    io.imsave("reconstruction.jpg", Mhat.reshape(600, 600, 3).astype('uint8'))

if __name__=='__main__':
    main()
