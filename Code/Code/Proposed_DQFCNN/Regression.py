import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def classify(Fea,tr,label):




    X_train, X_test, y_train, y_test = train_test_split(Fea, label, train_size=tr,random_state=1)

    ''' X_train=X_train[0]
    X_test=X_test[0]
    X_train=np.resize(X_train,(len(X_train),10))
    X_test=np.resize(X_test,(len(X_test),10))
    y_train=np.resize(y_train,(len(X_train),10))'''

    reg=LinearRegression().fit(X_train, y_train)
    Prediction=reg.predict(Fea)


    return np.array(Prediction).flatten()


