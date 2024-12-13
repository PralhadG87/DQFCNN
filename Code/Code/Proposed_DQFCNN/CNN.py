from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


def classify(Data,Label,tr):

    X_train, X_test, y_train, y_test = train_test_split(Data, Label, train_size = tr, random_state = 42)


    # Variables
    INPUT_SHAPE = (32, 32, 3)
    FILTER1_SIZE = 32
    FILTER2_SIZE = 64
    FILTER_SHAPE = (3, 3)
    POOL_SHAPE = (2, 2)
    FULLY_CONNECT_NUM = 128
    NUM_CLASSES = 10


    # Model architecture implementation
    opt = tf.keras.optimizers.legacy.RMSprop()
    model = Sequential()
    model.add(Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Flatten())
    model.add(Dense(FULLY_CONNECT_NUM, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(opt, loss='mse')


    X_train=np.resize(X_train,(len(X_train),32,32,3))
    X_test=np.resize(X_test,(len(X_test),32,32,3))

    model.fit(X_train,y_train,batch_size=100, epochs=10,verbose=0)
    weight = opt.get_weights()


    Pred=model.predict(X_test)


    return np.array(Pred).flatten(),weight
