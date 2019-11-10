from helpers_V2 import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys
import os

def CreateModel(modelName,filename,param):
    X_train,Y_train=ReadData(filename,True)
    num=os.path.splitext(filename)[0][4:]
    #print(num)
    SaveModel(modelName,modelName+str(param)+'_'+num,X_train,Y_train)

if __name__ == "__main__":
    modelName=sys.argv[1]
    loopVal=int(sys.argv[2])
    numOfSamples=int(sys.argv[3])
    param=int(sys.argv[4])
    fileSuffix=str(loopVal*loopVal)+"_"+str(numOfSamples)+".csv"
    dataFileName="file"+fileSuffix
    CreateModel(modelName,dataFileName,param)

