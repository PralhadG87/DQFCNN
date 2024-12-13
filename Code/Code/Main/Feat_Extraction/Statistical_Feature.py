import cv2
import numpy as np
import scipy
from Main import read

def Stat_Fea(path):
    Mean_,Variance_,Kurtosis_,Skew_,std_Deviation_=[],[],[],[],[]
    Img_path=read.image(path)

    for i in range(len(Img_path[0:2000])):
        print("Stat Fea :", i)
        img = cv2.imread(Img_path[i])

        Mean= img.mean()
        Mean_.append(Mean)

        Variance=np.var(img)
        Variance_.append(Variance)

        std_Deviation=np.std(img)
        std_Deviation_.append(std_Deviation)

        kur = scipy.stats.kurtosis(img)
        Kurtosis_.append(np.nan_to_num(kur[0]))

        sk = scipy.stats.skew(img)
        Skew_.append(np.nan_to_num(sk[0]))

        Stat_Fea=np.column_stack((Mean_,Variance_,Kurtosis_,Skew_,std_Deviation_))


    return Stat_Fea



