from helpers_V2 import *
from sys import *
import matplotlib.pyplot as plt
#from datetime import datetime
import random
import time

#Trying ROOT histogram
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double
l=0.5
a=0
b=0
def Process(modelName,loopVal,numOfSamples,numOfProfiles,param):
    modelFileName=GetModelFileName_Param(modelName,loopVal,numOfSamples,param)
    print("===================================")
    print("Model file read : "+modelFileName)
    print("===================================")
    #model=ReadModel(modelFileName)
    X_train,Y_train=ReadData("file100_1000.csv",True)
    model=Classifier["KERAS"]
    model.fit(X_train, Y_train, epochs = num_epoch, batch_size = 64)# ,validation_split=0.20)
    rootFileName=GetROOTFileName(modelName,loopVal,numOfSamples)
    hfile = TFile(rootFileName, 'RECREATE', 'Demo ROOT file with histograms' )
    c1=TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
    hpa=TH1F( 'hpa', 'This is the deviation distribution of A', 100, -1.0, 1.0 )
    hpb=TH1F( 'hpb', 'This is the deviation distribution of B', 100, -1.0, 1.0 )
    hpEpsA=TH1F( 'hpEpsA', 'This is the deviation distribution of A satisfying the Score', 100, -1.0, 1.0 )
    hpEpsB=TH1F( 'hpEpsB', 'This is the deviation distribution of B satisfying the Score', 100, -1.0, 1.0 )
    dictFileName=GetDictFileName(loopVal,numOfSamples)
    classDict=FillDict(dictFileName);

    list_a_act=[]
    list_a_pred=[]
    list_b_act=[]
    list_b_pred=[]

    list_a_predB0=[]
    list_b_predB0=[]
    numOfProfiles=int(numOfProfiles)
    for ite in range(numOfProfiles):
        supList=[]#.clear()
        print("========= Profile Num : "+str(ite)+" =========")
        profSamples=50
        z=np.empty((profSamples,1))
        for i in range(1):
            #a=0.355469
            a=np.random.uniform(0.1,0.5)
            #a=0.251002216634 #np.random.uniform(0.1,0.5)
            for k in range(1):
                #b=0.94639541#a+0.0000000001+np.random.uniform(0.3,0.9)
                b=a+0.0000000001+np.random.uniform(0.3,0.9)
                #b=np.random.uniform(a+0.001,1.5)
                for m in range(1):
                    j= 334 #432.1241038 #np.random.uniform(1,1000)
                    if(b>a):
                        Fo=getFo(a,b,l)
                        Bo=getBo(j,a,Fo)
                        z=np.linspace(-3*l/2,3*l/2,profSamples)
                        F=getF(z,a,l,b)
                        print(z.shape)
                        print(F.shape)  
                        BList=[]
                        for ind in range(len(z)):
                            Btest=getB(j,a,F[ind])
                            BList.append(Btest)
                            zztest=z[ind]
                            jjtest=j
                            #identest=i+10*k
                            subList=[Btest,zztest,jjtest,52]#,identest]
                            supList.append(subList)
                        BArray=np.array(BList)
        supListArray=np.array(supList)
        X_test=supListArray[:,0:3]
        Y_test=supListArray[:,3]
        magClass,X_pred=DoClassficationKeras(model,X_test,Y_test)
        print("MagClass : "+str(magClass))

        #classBasedOnB0=X_pred[profSamples/2]

        #based on Max count in histogram
        aPred,bPred,lPred=classDict[magClass]

        #based on B0
        #aPredB0,bPredB0,lPredB0=classDict[classBasedOnB0]

        list_a_act.append(a)
        list_b_act.append(b)
        list_a_pred.append(aPred)
        list_b_pred.append(bPred)

        #list_a_predB0.append(aPredB0)
        #list_b_predB0.append(bPredB0)
        
        diffA=a-aPred
        diffB=b-bPred
        diffL=l-lPred
        hpa.Fill(diffA)
        hpb.Fill(diffB)
        if((np.absolute(diffA) <= eps) and (np.absolute(diffB) <= eps)):
            hpEpsA.Fill(diffA)
            hpEpsB.Fill(diffB)

        print("a :"+str(a)+" b :"+str(b))
        #PlotProfile(BArray,z,X_pred)


    arr_a_actual=np.array(list_a_act)
    arr_b_actual=np.array(list_b_act)
    arr_a_predict=np.array(list_a_pred)
    arr_b_predict=np.array(list_b_pred)

    #arr_a_predictB0=np.array(list_a_predB0)
    #arr_b_predictB0=np.array(list_b_predB0)

    #based on max count in histogram
    score=GetScore(arr_a_actual,arr_a_predict,arr_b_actual,arr_b_predict)

    #based on B0
    #scoreB0=GetScore(arr_a_actual,arr_a_predictB0,arr_b_actual,arr_b_predictB0)

    hpa.Draw()
    hpb.Draw()
    hpEpsA.Draw()
    hpEpsB.Draw()
    c1.Modified()
    c1.Update()
    hfile.Write()
    return score#,scoreB0


if __name__=="__main__":
    score=Process(argv[1],argv[2],argv[3],argv[4],argv[5])
    print("Score from main : Based On Histogram : "+str(score[0])+" : Based On B0 : "+str(score[1]))






























