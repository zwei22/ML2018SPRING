import numpy as np
from keras.models import load_model
import sys, os 
from util import DataManager

def write(filename, output):
    with open(filename, "w") as of:
        of.write('id,label\n')
        for i in range(output.shape[0]):
            of.write('{0},{1}\n'.format(i, output[i]))

def predict(x_test, path_pfx):
    categorical_crossentropy = True

    output = np.zeros(len(x_test)) 
    model = load_model('model/model8326.h5')
    result = model.predict(x_test).squeeze()
    if categorical_crossentropy:
        output = result.argmax(axis=1)
    else:
        output = result.round()
    print(output.shape)
    return output.astype('int16')

def main():
    path_pfx = ''
    max_len = 37

    dm = DataManager()
    dm.add_data('test', os.path.join(sys.argv[1]), False, True)
    print(len(dm.data['test'][0]))
    dm.preprocessing()
    dm.load_word2vec(os.path.join(path_pfx, 'model/word2vec'))
    #dm.load_tokenizer(os.path.join(path_pfx, 'token.pkl'))
    dm.to_sequence(max_len, use_pretrain=True)
    result = predict(dm.data['test'][0], path_pfx)
    write(sys.argv[2], result)
    print('finished')
    
if __name__=='__main__':
    main()
