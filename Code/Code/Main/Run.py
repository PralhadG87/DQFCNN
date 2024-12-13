import Preprocessing
import Image_Agumentation
import Fea_Ext
import Proposed_DQFCNN.CNN
import Proposed_DQFCNN.Features
import Proposed_DQFCNN.Regression
import Proposed_DQFCNN.DQNN
import BNDFC.SVM
import PDHF.DRN
import DNN.Deep_NN
import CNN_RNN.RCNN


def callmain(itr,tr):
    ACC,TPR,FPR=[],[],[]
    Input_Img='Dataset/*/*'

    Preprocessing.images(Input_Img)  # preprocessing Image by using kalman Filter
    #bird segmentation using Fast Fuzzy C-Means Clustering
    seg_img_path='Processed/Segmentation/*'  # segmented Img is stored in this path
    #image augmentation (resize, rotation, translation, random erasing)
    Rotation,Resizing,Translation,Random_Erasing=Image_Agumentation.Images(seg_img_path)
    # Feature Extraction (LVP, Statistical features)
    Feature,Label=Fea_Ext.callmain(Rotation,Resizing,Translation,Random_Erasing)
    #Bird species classification using proposed SDCNN, which is the new deep learning architecture based on the  integration of SpinalNet, DCNN by modifying the layers of the networks

    l1,Weight=Proposed_DQFCNN.CNN.classify(Feature, Label, tr)

    F1,Lab1=Proposed_DQFCNN.Features.callmain(Feature, Weight, l1, Label)

    l2=Proposed_DQFCNN.Regression.classify(F1, tr, Lab1)
    #-------------Proposed_DQFCNN-----------------
    Proposed_DQFCNN.DQNN.qdnn_classify(F1, l2, Lab1, itr, tr, ACC, TPR, FPR)
   

    return ACC,TPR,FPR


