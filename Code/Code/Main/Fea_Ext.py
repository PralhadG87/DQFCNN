from Feat_Extraction import LVP
from Feat_Extraction import Statistical_Feature
import numpy as np
import glob
import pandas as pd

def fea(Rotation_Img,Resizing_Img,Translation_Img,Random_Erasing_Img):


    Rotation_Fea=LVP.Lvp_fea(Rotation_Img)
    Resizing_Fea=LVP.Lvp_fea(Resizing_Img)
    Trans_Fea=LVP.Lvp_fea(Translation_Img)
    Random_Er_Fea=LVP.Lvp_fea(Random_Erasing_Img)

    LVP_Fea=np.column_stack((Rotation_Fea,Resizing_Fea,Trans_Fea,Random_Er_Fea))
    np.savetxt("Processed/Extracted_Features/LVP_Feature.csv", LVP_Fea, delimiter=',', fmt='%s')

    S1=Statistical_Feature.Stat_Fea(Rotation_Img)
    S2=Statistical_Feature.Stat_Fea(Resizing_Img)
    S3=Statistical_Feature.Stat_Fea(Translation_Img)
    S4=Statistical_Feature.Stat_Fea(Random_Erasing_Img)

    SF=np.column_stack((S1,S2,S3,S4))
    np.savetxt("Processed/Extracted_Features/Statistical_Feature.csv", SF, delimiter=',', fmt='%s')



def callmain(Rotation_Img,Resizing_Img,Translation_Img,Random_Erasing_Img):

    #fea(Rotation_Img,Resizing_Img,Translation_Img,Random_Erasing_Img)

    # setting the path for joining multiple files
    path = "Processed/Extracted_Features/*"
    # list of merged files returned
    files = glob.glob(path)
    #print()
    # joining files with concat and read_csv
    df = pd.concat(map(pd.read_csv, files), axis=1)

    feature=np.array(df)

    Label=pd.read_csv('Processed/Label.csv')
    Label=np.array(Label)
    Label=Label[0:1999]

    return feature,Label
