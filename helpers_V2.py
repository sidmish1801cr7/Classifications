import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
import sys
import time

from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F

from MyDict import *


muo=4*np.pi*pow(10,-7)
eps=0.1

num_neighbours=10
num_estimators=10
Classifier={
        "KNN": KNeighborsClassifier(n_neighbors=num_neighbours),
        "DT": tree.DecisionTreeClassifier(),
        "RF": RandomForestClassifier(n_estimators=num_estimators)
        }

def GetCSVSuffix(loopVal,numOfSamples):
    loopVal=int(loopVal)
    numOfSamples=int(numOfSamples)
    suffix=str(loopVal*loopVal)+"_"+str(numOfSamples)+".csv"
    return suffix

def GetModelSuffix(loopVal,numOfSamples):
    loopVal=int(loopVal)
    numOfSamples=int(numOfSamples)
    suffix=str(loopVal*loopVal)+"_"+str(numOfSamples)+".sav"
    return suffix

def GetROOTSuffix(loopVal,numOfSamples):
    loopVal=int(loopVal)
    numOfSamples=int(numOfSamples)
    suffix=str(loopVal*loopVal)+"_"+str(numOfSamples)+".root"
    return suffix

def GetModelFileName(modelName,loopVal,numOfSamples):
    loopVal=int(loopVal)
    numOfSamples=int(numOfSamples)
    modelFileName="model_"+modelName+'_'+GetModelSuffix(loopVal,numOfSamples)
    return modelFileName

def GetModelFileName_Param(modelName,loopVal,numOfSamples,param):
    loopVal=int(loopVal)
    param=str(param)
    numOfSamples=int(numOfSamples)
    modelFileName="model_"+modelName+param+'_'+GetModelSuffix(loopVal,numOfSamples)
    return modelFileName


def GetDataFileName(loopVal,numOfSamples):
    loopVal=int(loopVal)
    numOfSamples=int(numOfSamples)
    dataFileName="file"+GetCSVSuffix(loopVal,numOfSamples)
    return dataFileName

def GetTrainingDataFileName(loopVal,numOfSamples):
    trainingFileName="training_"+GetDataFileName(loopVal,numOfSamples)

def GetTestDataFileName(loopVal,numOfSamples):
    trainingFileName="test_"+GetDataFileName(loopVal,numOfSamples)

def GetTrainingModelFileName(modelName,loopVal,numOfSamples):
    trainingFileName="training_"+GetModelFileName(modelName,loopVal,numOfSamples)

#def SplitDataAndGenerateTrainingModel(modelName,loopVal,numOfSamples):
def SplitAndStoreData(modelName,loopVal,numOfSamples):
    dataFileName=GetDataFileName(loopVal,numOfSamples)
    trainingDataFileName=GetTrainingDataFileName(loopVal,numOfSamples)
    testDataFileName=GetTestDataFileName(loopVal,numOfSamples)
    X_train, X_test, Y_train, Y_test=SplitData(dataFileName,0.2)
    train_array=np.concatenate((X_train,Y_train),axis=1)
    test_array=np.concatenate((X_test,Y_test),axis=1)
    np.savetxt(trainingDataFileName,train_array,delimiter=" ")
    np.savetxt(testDataFileName,test_array,delimiter=" ")

def TuneIt(modelName,loopVal,numOfSamples):
    dataFileName=GetDataFileName(loopVal,numOfSamples)
    print("=== DataFileName : "+dataFileName+" ========")
    X_train, X_test, Y_train, Y_test=SplitData(dataFileName,0.2)
    #print(Y_test)
    print(X_train.shape)
    print(X_test.shape)
    #NearestNeighbours(X_train,Y_train,X_test,Y_test,writeToFile=True)
    clf=Classifier[modelName]
    clf = clf.fit(X_train,Y_train )
    score = clf.score(X_test, Y_test)
    print(score)


    #Filename of training Model
    #trainingModelFileName=GetTrainingModelFileName(modelName,loopVal,numOfSamples)
    
    #Assuming data is written to the above mentioned file names

    #X_train,Y_train=ReadData(trainingDataFileName,True)
    #SaveModel(modelName,trainingModelFileName,X_train,Y_train)

def SplitData(filename,test_data=0.2):
    df=pd.read_csv(filename,delim_whitespace=True,names=('B','z','j','iden'))
    X=df[['B','z','j']]
    Y=df[['iden']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_data)
    return X_train, X_test, Y_train, Y_test

def GetScore(arr_a_actual,arr_a_predict,arr_b_actual,arr_b_predict):
    counter=0
    leng=arr_a_actual.shape[0]
    for i in range(leng):
        valid=(np.absolute(arr_a_actual[i]-arr_a_predict[i]) < eps ) and (np.absolute(arr_b_actual[i]-arr_b_predict[i]) < eps )
        if(valid):
            counter=counter+1

    print("counter : "+str(counter))
    print("leng : "+str(leng))
    return float(counter)/leng




def GetDictFileName(loopVal,numOfSamples):
    loopVal=int(loopVal)
    numOfSamples=int(numOfSamples)
    dictFileName="out_prac_"+GetCSVSuffix(loopVal,numOfSamples)
    return dictFileName

def GetROOTFileName(modelName,loopVal,numOfSamples):
    loopVal=int(loopVal)
    numOfSamples=int(numOfSamples)
    rootFileName=modelName+'_'+GetROOTSuffix(loopVal,numOfSamples)
    return rootFileName


def SaveModel(modelName,modelFileName,X_train,Y_train):
    #print("========== %s Classifier =========" % whoami())
    clf=Classifier[modelName] #KNeighborsClassifier(n_neighbors=num_neighbours)
    clf = clf.fit(X_train,Y_train )
    filename='model_'+modelFileName+'.sav'
    pickle.dump(clf,open(filename,'wb'))

def GetFileName(fullFileName):
    filename=os.path.splitext(filename)[0]
    return filename


def ReadModel(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def whoami():
    import sys
    return sys._getframe(1).f_code.co_name

def FillRootHistogram(X_pred,numOfBins):
    maxHist    = TH1F( 'maxHist', '', numOfBins, 0, numOfBins )
    for val in X_pred:
        maxHist.Fill(val)
    binmax=maxHist.GetMaximumBin()
    magClass=round(maxHist.GetXaxis().GetBinCenter(binmax))
    magClass=magClass-1
    print("========== "+whoami()+" : Class : "+str(magClass) )
    return magClass

def FillMatPlotLibHistogram(X_pred,numOfBins):
    (n, bins, patches) = plt.hist(X_pred,bins=numOfBins)
    result = np.where(n == np.amax(n))
    toRound=bins[result[0]]
    if(len(result[0]) > 1):
        toRound=bins[result[0]][0]
    magClass=round(toRound)
    print("Max Value : "+str(n[result[0]])+" : Class : "+str(magClass))
    print("========== "+whoami()+" : Class : "+str(magClass) )
    return magClass

def PlotProfile(B,Z,X_pred):
    print("========== "+whoami()+" ===========")
    print(B.shape==Z.shape)
    #plt.scatter(Z,B)
    colormap = np.array(['r', 'g', 'b'])

    colList=[]
    for col in X_pred:
       colList.append(int(col))
    legends=colList
    categories = np.array(colList)
    #plt.scatter(Z, B, s=100,  c=colormap[categories])

    #import matplotlib.colors.Colormap
    #cmap=Colormap.Colormap("myMap",100)
    #print(cmap)
    print("X predicted mid value : "+str(X_pred[Z.shape[0]/2]))

    cmap=GetColorMap()
    plt.scatter(Z, B,c=cmap[categories])#, s=100,  c=124)
    plt.legend(legends)
    #plt.show()

'''
def GetScore(arr_a_actual,arr_a_predict,arr_b_actual,arr_b_predict):
    counter=0
    leng=arr_a_actual.shape[0]
    for i in range(leng):
        valid=(np.absolute(arr_a_actual[i]-arr_a_predict[i]) < eps ) and (np.absolute(arr_b_actual[i]-arr_b_predict[i]) < eps )
        if(valid):
            counter=counter+1

    print("counter : "+str(counter))
    print("leng : "+str(leng))
    return float(counter)/leng
'''
def Get_Mid_Score():
    print(" I am "+whoami())


def GetColorMap():
    from matplotlib import colors as mcolors
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    return np.array(sorted_names)


def DoClassfication(model,X_test,Y_test):
    #score = model.score(X_test, Y_test)
    #print(score)
    timePredictStart=time.time()
    X_pred=model.predict(X_test)
    timePredictEnd=time.time()
    print("======= Time taken for Prediction : "+str(timePredictEnd-timePredictStart)+" =======")
    #print(confusion_matrix(Y_test,X_pred))
    '''
    timeFillHistStart=time.time()
    print("============ Shape of X_pred : "+str(X_pred.shape)+" =============")
    (n, bins, patches) = plt.hist(X_pred,bins=400)# ,label="Classification Histogram")
    timeFillHistEnd=time.time()
    print("========== Time taken JUST TO FILL the histogram : "+str(timeFillHistEnd-timeFillHistStart)+" =========")

    timeFindMaxStart=time.time()
    result = np.where(n == np.amax(n))
    timeFindMaxEnd=time.time()
    print("===== Time taken to find max count bin : "+str(timeFindMaxEnd-timeFindMaxStart)+" =======")
    
    print("Shape-n : "+str(n.shape)+" : Shape bins : "+str(bins.shape))
    print("debuggin rounding error : "+str(bins[result[0]]))
    toRound=bins[result[0]]
    if(len(result[0]) > 1):
        toRound=bins[result[0]][0]

        

    #magClass=round(bins[result[0]])
    magClass=round(toRound)
    

    print("Max Value : "+str(n[result[0]])+" : Class : "+str(magClass))
    #plt.show()
    

    #print("===== Time taken to find max count bin : "+str(timeFindMaxEnd-timeFindMaxStart)+" =======")
    '''

    #print("**********************************")
    magClass=FillRootHistogram(X_pred,400)
    #FillMatPlotLibHistogram(X_pred,400)
    #print("**********************************")
    print("MagClass : "+str(magClass))
    magClass=round(magClass)
    return magClass,X_pred

    #n1,bins,patches=plt.hist(X_pred,bins=100)
    #print(np.argmax(n1))
    #nmax=np.max(n1)
    #arg_max=None
    #for j, _n in enumerate(n):
        #if _n==max:
            #arg_max=j
            #break
    #print b[arg_max]

def FillDict(filename):
    dict_obj = my_dictionary()
    fil=np.genfromtxt(filename)
    for row in fil:
        dict_obj.add(row[0],[row[1],row[2],row[3]])
    #print(dict_obj)
    return dict_obj

    '''
    dict1={}
    fil=np.genfromtxt(filename)
    lis=[]
    for row in fil:
        lis.append((row[0],[row[1],row[2],row[3]]))
        #dict1.update(row[0]:[row[1],row[2],row[3]])
    Dict=dict(list)
    print(Dict)
    '''

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
