import read
import numpy as np


Input_Img='Dataset/*/*'

Images=read.image(Input_Img)

Label=[]
for i in range(len(Images)):
    A=Images[i].split('\\')

    if A[1]=='Asian-Green-Bee-Eater':Label.append(0)
    elif A[1]=='Brown-Headed-Barbet':Label.append(1)
    elif A[1]=='Cattle-Egret':Label.append(2)
    elif A[1]=='Common-Kingfisher':Label.append(3)
    elif A[1]=='Common-Myna':Label.append(4)
    elif A[1]=='Common-Tailorbird':Label.append(5)
    elif A[1]=='Forest-Wagtail':Label.append(6)
    elif A[1]=='Gray-Wagtail':Label.append(7)
    elif A[1]=='Hoopoe':Label.append(8)
    elif A[1]=='House-Crow':Label.append(9)
    elif A[1]=='Indian-Peacock':Label.append(10)
    elif A[1]=='Indian-Roller':Label.append(11)



np.savetxt("Processed/Label.csv",Label,delimiter=',',fmt='%s')

print("Label :",len(Label))