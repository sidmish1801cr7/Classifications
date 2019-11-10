import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

muo=4*np.pi*pow(10,-7)


num_neighbours=3
Classifier={
        "KNN": KNeighborsClassifier(n_neighbors=num_neighbours),
        "DT": tree.DecisionTreeClassifier()
        }

def SaveModel(modelName,modelFileName,X_train,Y_train):
    #print("========== %s Classifier =========" % whoami())
    #if(Classifier[modelName]==1):
    clf=Classifier[modelName] #KNeighborsClassifier(n_neighbors=num_neighbours)
    #if(Classifier[modelName]==1):
     #   clf=tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,Y_train )
    filename='model_'+modelFileName+'.sav'
    pickle.dump(clf,open(filename,'wb'))

def ReadModel(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def DoClassfication(model,X_test,Y_test):
    score = model.score(X_test, Y_test)
    print(score)
    X_pred=model.predict(X_test)
    print(confusion_matrix(Y_test,X_pred))
    (n, bins, patches) = plt.hist(X_pred,bins=400,label="Classification Histogram")
    result = np.where(n == np.amax(n))
    print(n)
    print("Max Value : "+str(n[result[0]])+" : Class : "+str(result[0]))
    plt.show()
    #n1,bins,patches=plt.hist(X_pred,bins=100)
    #print(np.argmax(n1))
    #nmax=np.max(n1)
    #arg_max=None
    #for j, _n in enumerate(n):
        #if _n==max:
            #arg_max=j
            #break
    #print b[arg_max]
def csv_dict(filename):
    dict1={
            }
    fil=np.genfromtxt(filename)
    for row in fil:
        dict1.update(row[0]:[row[1],row[2],row[3]])
    print(dict1)




def getFo(a,b,l):
    alpha=b/a
    beta=(l/2)/a
    ret=muo*beta*np.log((alpha+np.sqrt(alpha**2+beta**2))/(1+np.sqrt(1+beta**2)))
    return ret
def getBo(j,a,Fo):
    return j*a*Fo
def getF(z,a,l,b):
    beta2=((l/2)+z)/a
    beta1=((l/2)-z)/a
    alpha=b/a
    ret=muo*(beta1*np.log((alpha+np.sqrt(alpha**2+beta1**2))/(1+np.sqrt(1+beta1**2)))+beta2*np.log((alpha+np.sqrt(alpha**2+beta2**2))/(1+np.sqrt(1+beta2**2))))
    return ret
def getB(j,a,F):
    return 0.5*j*a*F

def ReadData(filename,retY=True):
    df=pd.read_csv(filename,delim_whitespace=True,names=('B','z','j','iden'))
    X=df[['B','z','j']]
    if(retY):
        Y=df[['iden']]
        return X,Y
    else:
        return X
