import numpy as np
import io
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

def train(x_all, y_all, argumentation=False):
    val_rate = 0.1

    x_all = x_all.reshape(-1, 48, 48, 1);
    y_all = np_utils.to_categorical(y_all)
    #validation
    x_train, y_train, x_valid, y_valid = valid_split(x_all, y_all, val_rate)

    #data argumentation
    if argumentation:
        datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=30, zoom_range=0.2,
                shear_range=0.2, fill_mode='nearest')
        datagen.fit(x_train)
    #model
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))

    model.summary()
    #compiling
    adam = Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #check point
    filepath='model_best.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    #fitting params
    epochs = 50 
    batch_size = 256 
    samples_per_epoch = len(y_train)*8
    #fitting
    if argumentation:
        model.fit_generator(datagen.flow(x_train, y_train, batch_size),
                samples_per_epoch=samples_per_epoch, epochs=epochs,
                validation_data=(x_valid, y_valid),
                callbacks=callbacks_list)
    else:
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_valid, y_valid), 
                callbacks=callbacks_list)

    score = model.evaluate(x_all, y_all)
    print(score)
    model.save('model.h5')

def valid_split(x_train, y_train, val_rate):
    n = int(len(y_train)*(1-val_rate))
    x_valid = x_train[n:]
    y_valid = y_train[n:]
    x_train = x_train[:n]
    y_train = y_train[:n]
    return x_train, y_train, x_valid, y_valid

def normalization(x_train):
    x_train /= 255
    return x_train

def read(filename):
    data = io.StringIO(open(filename).read().replace(',',' '))
    all_train = np.genfromtxt(data, delimiter=' ',  skip_header=1)
    y_train = all_train[:, 0] 
    x_train = all_train[:, 1:] 
    #np.save('x_train.npy', x_train)
    #np.save('y_train.npy', y_train)
    return x_train, y_train
    
def read_from_npy():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    return x_train, y_train
    p = np.random.permutation(len(y_train))
    return x_train[p], y_train[p]

def main():
    x_train, y_train = read(sys.argv[1])
    #x_train, y_train = read_from_npy()
    x_train = normalization(x_train)
    train(x_train, y_train, argumentation=True)
    #train(x_train, y_train)

if __name__=="__main__":
    main()
