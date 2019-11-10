from helpers_V2 import *
from sys import *
import matplotlib.pyplot as plt
#from datetime import datetime
import random
import time

#Trying ROOT histogram
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F, TF1
from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double
l=0.5
a=0
b=0
'''
listDiffA=[]
listDiffB=[]
listDiffL=[]

modelName=argv[1]
loopVal=argv[2]
numOfSamples=argv[3]
numOfProfiles=int(argv[4])

modelFileName=GetModelFileName(modelName,loopVal,numOfSamples)
print("===================================")
print("Model file read : "+modelFileName)
print("===================================")
model=ReadModel(modelFileName)

rootFileName=GetROOTFileName(modelName,loopVal,numOfSamples)
hfile = TFile(rootFileName, 'RECREATE', 'Demo ROOT file with histograms' )
c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
hpa    = TH1F( 'hpa', 'This is the deviation distribution of A', 100, -1.0, 1.0 )
hpb    = TH1F( 'hpb', 'This is the deviation distribution of B', 100, -1.0, 1.0 )

dictFileName=GetDictFileName(loopVal,numOfSamples)
classDict=FillDict(dictFileName);

list_a_act=[]
list_a_pred=[]
list_b_act=[]
list_b_pred=[]
for ite in range(numOfProfiles):
    supList=[]#.clear()
    print("========= Profile Num : "+str(ite)+" =========")
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
                    z=np.linspace(-3*l/2,3*l/2,500)
                    F=getF(z,a,l,b)
                    print(z.shape)
                    print(F.shape)
                    for ind in range(len(z)):
                        Btest=getB(j,a,F[ind])
                        zztest=z[ind]
                        jjtest=j
                        #identest=i+10*k
                        subList=[Btest,zztest,jjtest,52]#,identest]
                        supList.append(subList)
    supListArray=np.array(supList)
    print(supListArray.shape)
    #np.savetxt('test_data_temp.csv',supListArray,delimiter=' ')

    #X_test,Y_test=ReadData('test_data_temp.csv',True)
    X_test=supListArray[:,0:3]
    Y_test=supListArray[:,3]
    magClass=DoClassfication(model,X_test,Y_test)
    #dictFileName=GetDictFileName(loopVal,numOfSamples) #argv[2] #"Something depending on argv[1]"
    #classDict=FillDict(dictFileName);
    aPred,bPred,lPred=classDict[magClass]
    print("===============================================")
    print("a : "+str(a)+" : b : "+str(b)+" : J : "+str(j))
    list_a_act.append(a)
    list_b_act.append(b)
    list_a_pred.append(aPred)
    list_b_pred.append(bPred)
    
    diffA=a-aPred
    diffB=b-bPred
    diffL=l-lPred
    hpa.Fill(diffA)
    hpb.Fill(diffB)
    print("DiffA : "+str(diffA)+" : diffB : "+str(diffB))

    #listDiffA.append(diffA);
    #listDiffB.append(diffB);
    #listDiffL.append(diffL);


#plt.style.use('seaborn-deep')
#diffAarray=np.asarray(listDiffA,dtype='float')
#print(diffAarray)
#diffBarray=np.array(listDiffB)
#diffLarray=np.array(listDiffL)
#plt.hist([diffAarray, diffBarray], label=['a', 'b'])
#binss=np.linspace(-0.30,0.30,20)
#plt.hist(diffAarray,normed=True,bins=10)
#plt.hist(diffBarray)
#plt.legend(loc='upper right')
#plt.show()


#using our Scoring routine
arr_a_actual=np.array(list_a_act)
arr_b_actual=np.array(list_b_act)
arr_a_predict=np.array(list_a_pred)
arr_b_predict=np.array(list_b_pred)
score=GetScore(arr_a_actual,arr_a_predict,arr_b_actual,arr_b_predict)
print("==================================")
print("Calculated Score : "+str(score))
print("==================================")

hpa.Draw()
hpb.Draw()
c1.Modified()
c1.Update()

hfile.Write()
'''
def Process(modelName,loopVal,numOfSamples,numOfProfiles,param):
    modelFileName=GetModelFileName_Param(modelName,loopVal,numOfSamples,param)
    print("===================================")
    print("Model file read : "+modelFileName)
    print("===================================")
    model=ReadModel(modelFileName)
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
        profSamples=1001
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
        magClass,X_pred=DoClassfication(model,X_test,Y_test)

        classBasedOnB0=X_pred[profSamples/2]

        #based on Max count in histogram
        aPred,bPred,lPred=classDict[magClass]

        #based on B0
        aPredB0,bPredB0,lPredB0=classDict[classBasedOnB0]

        list_a_act.append(a)
        list_b_act.append(b)
        list_a_pred.append(aPred)
        list_b_pred.append(bPred)

        list_a_predB0.append(aPredB0)
        list_b_predB0.append(bPredB0)
        
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

    arr_a_predictB0=np.array(list_a_predB0)
    arr_b_predictB0=np.array(list_b_predB0)

    #based on max count in histogram
    score=GetScore(arr_a_actual,arr_a_predict,arr_b_actual,arr_b_predict)

    #based on B0
    scoreB0=GetScore(arr_a_actual,arr_a_predictB0,arr_b_actual,arr_b_predictB0)

    #ft=hpa.Fit("gaus")
    g1 = TF1( 'g1', 'gaus')
    hpa.Fit( g1)#, 'R' )
    par1 = g1.GetParameters()
    par1err=g1.GetParErrors()

    g2 = TF1( 'g2', 'gaus')
    hpb.Fit( g2)#, 'R' )
    par2 = g2.GetParameters()
    par2err=g2.GetParErrors()
    '''

    hpEpsA.Fit( g1)#, 'R' )
    par3 = g1.GetParameters()

    hpEpsB.Fit( g1)#, 'R' )
    par4 = g1.GetParameters()
    '''
    print("====================================================================") 
    print("hpa : "+str(par1[0])+" : Mean : "+str(par1[1])+" : Sigma : "+str(par1[2]))
    print("hpaErrors : "+str(par1err[0])+" : Mean : "+str(par1err[1])+" : Sigma : "+str(par1err[2]))
    print("hpb : "+str(par2[0])+" : Mean : "+str(par2[1])+" : Sigma : "+str(par2[2]))
    print("hpaErrors : "+str(par2err[0])+" : Mean : "+str(par2err[1])+" : Sigma : "+str(par2err[2]))
    '''
    print("hpEpsA : "+str(par3[0])+" : Mean : "+str(par3[1])+" : Sigma : "+str(par3[2]))
    print("hpEpsB : "+str(par4[0])+" : Mean : "+str(par4[1])+" : Sigma : "+str(par4[2]))
    '''

    hpa.Draw()
    hpb.Draw()
    hpEpsA.Draw()
    hpEpsB.Draw()
    c1.Modified()
    c1.Update()
    hfile.Write()
    return score,scoreB0


if __name__=="__main__":
    score=Process(argv[1],argv[2],argv[3],argv[4],argv[5])
    print("Score from main : Based On Histogram : "+str(score[0])+" : Based On B0 : "+str(score[1]))






























