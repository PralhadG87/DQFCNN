from skimage import  segmentation, color
from PIL import Image
import cv2
import numpy as np
from Main import read

def Lvp_fea(Img_path):

    Img_list=read.image(Img_path)


    LVP_F=[]
    for ii in range(len(Img_list[0:2000])):
        print("ii:", ii)

        img=cv2.imread(Img_list[ii])
        #print("img shape :", img.shape)
        '''if img.shape > (500,500,3):
            img.shape=(500,500,3)'''

        #print("img shape :",img.shape)

        labels1 = segmentation.slic(img, compactness=10, n_segments=500,
                                start_label=1)
        out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
        out1 = Image.fromarray(out1, 'RGB')


        out1=np.uint8(out1)

        histograms = cv2.calcHist([out1], [0], None, [100], [0, 100])  # calculating feature vector for the image
        histograms = np.array(histograms)
        histograms = histograms.flatten()
        LVP_F.append(histograms)


    return LVP_F





