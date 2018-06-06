import os, sys
import numpy as np
from keras import regularizers
from keras.models import Sequential 
from keras.layers import GRU, LSTM, Dense, Dropout, Bidirectional, Embedding, SpatialDropout1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from util import DataManager

def train(dm, voc_size, max_len, emb_mat=None):
    # model
    mode = 'GRU'
    loss_function = 'categorical_crossentropy'
    bidirectional = False

    Embedding = dm.embedding_layer() 

    
    model = Sequential()
    model.add(Embedding)
    #model.add(SpatialDropout1D(0.3))

    if mode  == 'GRU':
        model.add(GRU(128, return_sequences=True,
                       dropout=0.2, recurrent_dropout=0.2))
        model.add(GRU(128, return_sequences=False,
                       dropout=0.2, recurrent_dropout=0.2))
    elif mode == 'LSTM':
        model.add(LSTM(128, return_sequences=False,
                        dropout=0.2, recurrent_dropout=0.2))
    if bidirectional:
        RNN_cell = Bidirectional(RNN_cell)

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    if loss_function == 'binary_crossentropy':
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(2, activation='softmax'))
            
    model.summary()
    # compile
    adam = Adam(5e-4)
    model.compile(loss=loss_function, optimizer=adam, metrics=[ 'accuracy',])
    # fit
    val_ratio = 0.1 
    batch_size = 256
    num_epoch = 30

    save_path = 'model/model.h5'
    checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    if loss_function == 'categorical_crossentropy':
        dm.to_category()
    (X,Y),(X_val,Y_val) = dm.split_data('train', val_ratio)

    earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')
    history = model.fit(X, Y, 
                        validation_data=(X_val, Y_val),
                        epochs=num_epoch, 
                        batch_size=batch_size,
                        callbacks=[checkpoint, earlystopping] )

def main():
    voc_size = None
    max_len = 39 
    path_pfx = ''
    dm = DataManager()
    dm.add_data('train', sys.argv[1])
    #dm.add_data('semi', os.path.join(path_pfx, 'training_nolabel.txt'), False)
    #dm.add_data('test', os.path.join(path_pfx, 'testing_data.txt'), False, True)
    dm.preprocessing()

    dm.load_word2vec(os.path.join(path_pfx, 'model/word2vec'))
    #dm.load_embedding_matrix(os.path.join(path_pfx, 'word2vec.wv.vectors.npy'))
    dm.to_sequence(max_len, use_pretrain=True)
    #dm.to_bow()

    print(max_len)

    #emb_mat =  dm.get_embedding_matrix()
    emb_mat = None

    train(dm, voc_size=voc_size, max_len=max_len, emb_mat=emb_mat)
    


if __name__=='__main__':
    main()

