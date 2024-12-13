import cv2
import numpy as np
import read


def paint_canvas(tp):

    for i in range(len(measured) - 1): cv2.line(dr_frame, measured[i], measured[i + 1], (0, 100, int(tp)))
    for i in range(len(predicted) - 1): cv2.line(dr_frame, predicted[i], predicted[i + 1], (0, 0, int(tp)))
    return dr_frame

def preprocess(path):
    global dr_frame, measured, predicted


    Images=read.image(path)
    measured = []
    predicted = []
    for i in range(len(Images)):
        print("i :",Images[i])

        measured = []
        predicted = []
        img = cv2.imread(Images[i])
        dr_frame = img
        mp = np.array((2, 1), np.float32)


        kalman_fil = cv2.KalmanFilter(4, 2)
        kalman_fil.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman_fil.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman_fil.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                              np.float32) * 0.03
        kalman_fil.correct(mp)

        tp = kalman_fil.predict()
        predicted = paint_canvas(tp)

        cv2.imwrite('Processed//Preprocessed//'+str(i)+'.jpg',predicted)


def images(path):
    #preprocess(path)

    pre_img_path='Processed//Preprocessed//*'

    return pre_img_path