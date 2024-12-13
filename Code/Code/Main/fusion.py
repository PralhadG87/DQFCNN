import random

def callmain(Fea,L2,Label):

    F=[]
    for i in range(len(Fea)):
        temp=[]
        for j in range(len(Fea[i])):

            temp.append(Fea[i][j]*L2[i])
        F.append(temp)
    L=[]
    for i in range(len(F)):
        L.append(Label[i])


    return F,L




