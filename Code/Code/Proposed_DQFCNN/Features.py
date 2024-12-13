

def callmain(Features,weight,l1,Label):

    LVP_f=Features[0:400]

    SF_f=Features[401:-1]

    y,y1=[],[]

    for i in range(len(LVP_f)):

        temp=[]
        for j in range(len(LVP_f[i])):

            temp.append(LVP_f[i][j]*weight[0])
        y.append(temp)


    for i in range(len(SF_f)):

        temp1=[]
        for j in range(len(SF_f[i])):

            temp1.append(SF_f[i][j]*weight[0])

        y1.append(temp1)
    y2=l1


    #----------------Applying  Fractional Concept----------------------------
    F=[]
    h=10 # Constant Number

    for i in range(len(y)):

        tem2=[]
        for j in range(len(y[i])):

            tem2.append(h*y[i][j]+(1/2)*h*(y1[i][j])+(1/6)*(1-h)*y2[i])
        F.append(tem2)
    Label_=[]
    for i in range(len(F)):
        Label_.append(Label[i])

    return F,Label_