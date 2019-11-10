from helpers import *
from sys import *
supList=[]
l=0.5
a=0
b=0
for i in range(1):
    #a=0.355469
    a=np.random.uniform(0.1,0.5)
    #a=0.251002216634 #np.random.uniform(0.1,0.5)
    for k in range(1):
        #b=0.94639541#a+0.0000000001+np.random.uniform(0.3,0.9)
        b=a+0.0000000001+np.random.uniform(0.3,0.9)
        for m in range(1):
            j= 334 #432.1241038 #np.random.uniform(1,1000)
            if(b>a):
                Fo=getFo(a,b,l)
                Bo=getBo(j,a,Fo)
                z=np.linspace(-3*l/2,3*l/2,1000)
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
np.savetxt('test_data_5.csv',supListArray,delimiter=' ')

model=ReadModel(argv[1])
X_test,Y_test=ReadData('test_data_5.csv',True)
DoClassfication(model,X_test,Y_test)


print("===============================================")
print("a : "+str(a)+" : b : "+str(b)+" : J : "+str(j))


