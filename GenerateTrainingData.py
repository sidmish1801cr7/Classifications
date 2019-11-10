#import ROOT
from helpers import *
from sys import *
'''
try:
    import numpy as np
except:
    print("Failed to import numpy")
    exit()
muo=4*np.pi*pow(10,-7)
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
'''
out_sublist=[]
out_suplist=[]
subList=[]
supList=[]
l=0.5
j=np.linspace(1,1000,10)
for i in range(40):
    a=np.random.uniform(0.1,0.5)
    for k in range(40):
        iden=10*i+k
        b=a+0.0000000001+np.random.uniform(0.3,0.9)
        for m in range(10):
            #j=np.random.uniform(1,1000)
            if(b>a):
                Fo=getFo(a,b,l)
                Bo=getBo(j[m],a,Fo)
                z=np.linspace(-3*l/2,3*l/2,argv[2])
                F=getF(z,a,l,b)
                #iden=10*i+k
                #print("Raman "+str(iden)+ " : a : "+str(a)+" : b : "+str(b)+" : j :"+str(j))
                #out_sublist=[iden,a,b]
                #out_suplist.append(out_sublist)
                for ind in range(len(z)):
                    B=getB(j[m],a,F[ind])
                    zz=z[ind]
                    jj=j[m]
                    subList=[B,zz,jj,iden]
                    supList.append(subList)
        out_sublist=[iden,a,b]
        out_suplist.append(out_sublist)
supListArray=np.array(supList)
np.savetxt(argv[1],supListArray,delimiter=' ')
out_suplist=np.array(out_suplist)
np.savetxt(argv[3],out_suplist,delimiter=' ')


















#function to make the tree
#def make_tree():
    #l=0.5
    #root_file=ROOT.TFile("db_l_1.root","RECREATE")
    #tree=ROOT.TTree("tree","tutorial")
    #B=np.empty((1),dtype="float32")
    #zz=np.empty((1),dtype="float32")
    #jj=np.empty((1),dtype="float32")
    #iden=np.empty((1),dtype="i")
    #tree.Branch("B",B,"B/F")
    #tree.Branch("zz",zz,"zz/F")
    #tree.Branch("jj",jj,"jj/F")
    #tree.Branch("iden",iden,"x4/I")
    #for i in range(10):
        #a=np.random.uniform(0.1,0.5)
        #for k in range(10):
            #b=a+0.0000000001+np.random.uniform(0.3,0.9)
            #for m in range(10):
                #j=np.random.uniform(1,1000)
                #if (b>a):
                    #Fo=getFo(a,b,l)
                    #Bo=getBo(j,a,Fo)
                    #z=np.linspace(-3*l/2,3*l/2,1000)
                    #F=getF(z,a,l,b)
                    #for ind in range(len(z)):
                        #B=getB(j,a,F[ind])
                        #zz=z[ind]
                        #jj=j
                        #iden=i+10*k
                        #tree.Fill()
    #root_file.Write()
    #return (root_file), tree

#conversion to numpy array using multi-thread support
#ROOT.ROOT.EnableImplicitMT()
#creating a root file with a tree and branch "x1","x2","x3","x4"
#_, tree=make_tree()
#printing content of the by looping explicitly
#print("Tree content:\n{}\n".format(
    #np.asarray([[tree.x1, tree.x2, tree.x3, tree.x4] for event in tree])))
# Read-out full tree as numpy array
#array=tree.AsMatrix()
#print("Tree converted to a numpy array:\n{}\n".format(array))
## Get numpy array and according labels of the columns
#array,labels=tree.AsMatrix(return_labels=True)
#print("Return numpy array and labels:\n{}\n{}\n".format(labels,array))
# Apply numpy methods on the data
#print("Mean of the columns retrieved with a numpy method: {}\n".format(
    #np.mean(array,axis=0)))
## Convert the tree to a pandas.DataFrame
#try:
    #import pandas
#except:
    #print("Failed to import pandas.")
    #exit()
#data,columns=tree.AsMatrix(return_labels=True)
#df=pandas.DataFrame(data=data,columns=columns)
#print("Tree converted to a pandas.DataFrame:\n{}".format(df))




