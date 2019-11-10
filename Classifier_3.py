from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import pickle
#from readTree import *

#base learners for adaboost
from sklearn.svm import SVC
from sklearn import metrics

def KerasClassfier(X_train,Y_train,X_Test,Y_test,num_epoch=15):
    X_test=X_Test[:,0:6]

    print("========== %s Classifier =========" % whoami())
    from keras.models import Sequential
    from keras.layers import Dense,Dropout
    from keras.utils import to_categorical

    inputShape=X_train.shape[1]
    model = Sequential()
    model.add(Dense(8, input_shape=(inputShape,) , activation = 'relu'))
  #  model.add(Dropout(0.2))

    model.add(Dense(10, activation = 'relu'))
  #  model.add(Dropout(0.2))

    model.add(Dense(10, activation = 'relu'))
  #  model.add(Dropout(0.2))

    model.add(Dense(10, activation = 'relu'))
  #  model.add(Dropout(0.1))

    model.add(Dense(4, activation = 'softmax'))

    Y_train=to_categorical(Y_train)
    Y_test=to_categorical(Y_test)

    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
    model.fit(X_train, Y_train, epochs = num_epoch, batch_size = 64 ,validation_split=0.20)
    scores = model.evaluate(X_test, Y_test)
    Y_pred = model.predict(X_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
    print(matrix)

def MLP(X_train,Y_train,X_Test,Y_test,num_iter=100,act_func='tanh',writeToFile=False):
    X_test=X_Test[:,0:6]
    print("========== %s Classifier =========" % whoami())
    print(num_iter)
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(max_iter=num_iter,activation=act_func)
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))
    if(writeToFile):
      	WriteToFile(X_Test,X_pred,whoami())
    return clf

def RandomForest(X_train,Y_train,X_Test,Y_test,num_estimators=50,writeToFile=False):
    #X_test=X_Test[:,0:6]
    print("========== %s Classifier =========" % whoami())
    print(num_estimators)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=num_estimators)
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    print(X_pred)
    #print(X_pred)
    #score = clf.score(X_test, Y_test)
    #score = clf.predict_proba(X_test)

    #fpr,tpr,thres= roc_curve(Y_test, score)
    #roc_auc = auc(fpr, tpr)
    print("========= ROC ===========")
    print(X_Test.shape)
    #print(roc_auc)
    print(score)
    print(confusion_matrix(Y_test,X_pred))
    if(writeToFile):
      	WriteToFile(X_Test,X_pred,whoami())
    
    #PlotROCSci(clf,X_test,Y_test)
    
    #NoveltyDetection(X_test,X_pred)
    
    
     
    #Collected point which are predicted as Fe
    ''' 
    New Idea of remove outliers which are detected as iron.
    Able to remove some outlier, but results  are not
    upto expectation, hence commenting for the time being.
    
    But its working code
    '''
    
    '''
    counter = 0
    iron = 0
    supList=[]
    Fe_pred=[]
    for pred in X_pred:
        subList=[]
        if(pred==2):
           Fe_pred.append(2)
           iron = iron + 1
           for e in X_Test[counter]:
               subList.append(e) 
           supList.append(subList)
	counter = counter + 1
	
    dataArray=np.array(supList)
    feArray=np.array(Fe_pred)
    print("====== Iron Counter ======")
    print(iron)
    print(dataArray.shape)
    
    NoveltyDetection(dataArray,feArray,True)   
    '''

    return clf
    

def GradientBoosting(X_train,Y_train,X_Test,Y_test,num_estimators=100,writeToFile=False):
    X_test=X_Test[:,0:6]
    print("========== %s Classifier =========" % whoami())
    from sklearn.ensemble import  GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=num_estimators)
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))
    if(writeToFile):
      	WriteToFile(X_Test,X_pred,whoami())
    
    PlotROCSci(clf,X_test,Y_test)
    return clf


def DecisionTree(X_train,Y_train,X_test,Y_test,writeToFile=False):
    #X_test=X_Test[:,0:6]
    print("========== %s Classifier =========" % whoami())
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    print(X_pred)
    #score = clf.score(X_test, Y_test)
    #print(score)
    #print(confusion_matrix(Y_test,X_pred))
    #PlotROCSci(clf,X_test,Y_test)
    #if(writeToFile):
      	#WriteToFile(X_Test,X_pred,whoami())
    #return clf

def AdaBoost(X_train,Y_train,X_Test,Y_test,writeToFile=False):
    X_test=X_Test[:,0:6]
    print("========== %s Classifier =========" % whoami())
    from sklearn.ensemble import AdaBoostClassifier
    #clf = AdaBoostClassifier(    tree.DecisionTreeClassifier(max_depth=15),    n_estimators=100,    learning_rate=1.5,    algorithm="SAMME")
    svc=SVC(probability=True, kernel='linear')
    clf = AdaBoostClassifier( base_estimator=svc,    n_estimators=10,    learning_rate=1) #,    algorithm="SAMME")
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))
    if(writeToFile):
      	WriteToFile(X_Test,X_pred,whoami())

    PlotROCSci(clf,X_test,Y_test)
    return clf
    
def Bagging(X_train,Y_train,X_Test,Y_test,writeToFile=False):
    X_test=X_Test[:,0:6]
    print("========== %s Classifier =========" % whoami())
    from sklearn.ensemble import BaggingClassifier
    #clf = AdaBoostClassifier(    tree.DecisionTreeClassifier(max_depth=15),    n_estimators=100,    learning_rate=1.5,    algorithm="SAMME")
    svc=SVC(probability=True, kernel='linear')
    clf = BaggingClassifier( n_estimators=100) 
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    #print(confusion_matrix(Y_test,X_pred))
    #if(writeToFile):
      	#WriteToFile(X_Test,X_pred,whoami())

    #PlotROCSci(clf,X_test,Y_test)
    return clf

def LDA(X_train,Y_train,X_Test,Y_test,writeToFile=False):
    X_test=X_Test[:,0:6]
    print("========== %s Classifier =========" % whoami())
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))
    if(writeToFile):
      	WriteToFile(X_Test,X_pred,whoami())
    return clf
 
 
 def SaveModelNN(X_train,Y_train,num_neighbours=3,writeToFile=False):
    #X_test=X_Test[:,0:6]
    print("========== %s Classifier =========" % whoami())
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=num_neighbours)
    #Y_train=Y_train.reshape(X_train.shape[0])
    clf = clf.fit(X_train,Y_train )
    filename='model_NN.sav'
    pickle.dump(clf,open(filename,'wb'))
    #some time later
    loaded_model=pickle.load(open(filename,'rb'))
    X_pred=loaded_model.predict(X_test)
    print(X_pred)
    #print(X_pred.shape)
    #print(X_test.shape)
    score = loaded_model.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))
    plt.hist(X_pred)
    plt.show()
    #print(X_pred)
    #PlotROCSci(clf,X_test,Y_test)

def NearestNeighbours(X_train,Y_train,X_test,Y_test,num_neighbours=3,writeToFile=False):
    #X_test=X_Test[:,0:6]
    print("========== %s Classifier =========" % whoami())
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=num_neighbours)
    #Y_train=Y_train.reshape(X_train.shape[0])
    clf = clf.fit(X_train,Y_train )
    filename='model_1.sav'
    pickle.dump(clf,open(filename,'wb'))
    #some time later
    loaded_model=pickle.load(open(filename,'rb'))
    X_pred=loaded_model.predict(X_test)
    print(X_pred)
    #print(X_pred.shape)
    #print(X_test.shape)
    score = loaded_model.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))
    plt.hist(X_pred)
    plt.show()
    #print(X_pred)
    #PlotROCSci(clf,X_test,Y_test)
 
   '''
    if(writeToFile):
      	WriteToFile(X_test,X_pred,whoami())
    return clf
    '''
    
def Ensemble(X_train,Y_train,X_Test,Y_test,writeToFile=False):
    print("========== %s Classifier =========" % whoami())
    from sklearn.ensemble import VotingClassifier
    X_test=X_Test[:,0:6]
    rf=RandomForest(X_train,Y_train,X_test,Y_test,writeToFile=False,num_estimators=500)
    gb=GradientBoosting(X_train,Y_train,X_test,Y_test,writeToFile=True)
    dt=DecisionTree(X_train,Y_train,X_test,Y_test)
    lda=LDA(X_train,Y_train,X_test,Y_test)
    nn=NearestNeighbours(X_train,Y_train,X_test,Y_test)
    mlp=MLP(X_train,Y_train,X_test,Y_test,num_iter=200)

    estimators=[('gb', gb), ('dt', dt),  ('mlp', mlp),('rf',rf), ('nn',nn),('lda',lda)]
    clf = VotingClassifier(estimators, voting='soft')
    Y_train=Y_train.reshape(X_train.shape[0])
    clf = clf.fit(X_train,Y_train )
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print("---------------------------------")
    print(score)
    print(confusion_matrix(Y_test,X_pred))
    #PlotROC(clf,X_test,Y_test)
    PlotROCSci(clf,X_test,Y_test)
    if(writeToFile):
      	WriteToFile(X_Test,X_pred,whoami())
    return clf

def NoveltyDetection(X_Test,X_pred,writeToFile=False):
    X_test=X_Test[:,0:6]
    print("========== %s  =========" % whoami())
    from sklearn import svm
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    dataArr=load_training_data("TMVA_Fe.root",True)
    X=dataArr[:,0:6]
    Y=dataArr[:,9:10]
    X_train=X
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    print("====== N_train_error ========")
    print(n_error_train)
    print("====== N_test_error ========")
    print(n_error_test)
    #n_error_test = y_pred_test[y_pred_test == -1].size
  
    supList=[]
    counter = 0
    predFe=[]
    for pred in y_pred_test:
        subList=[]
        if(pred == 1):
            predFe.append(2)
	    for e in X_Test[counter]:
		subList.append(e)

	    supList.append(subList)
        counter = counter + 1
        
    dataArray=np.array(supList) 
    print("======== DataArray Shape =========")
    print(dataArray.shape)
    predArray=np.array(predFe)
    if(writeToFile):
        WriteToFile(dataArray,predArray,whoami())
    return clf
	
def whoami():
    import sys
    return sys._getframe(1).f_code.co_name
    
def WriteToFile(X_test,X_pred,functionName):
	supList=[]
	counter=0
	print(whoami())
	#print(X_test)
	print(X_test.shape)
	print(X_pred.shape)
	subList=X_test[0,6:9]
	#print(subList)
	
	for e in X_test:
		npSubList = e[6:9]
		subList=[]
		for n in npSubList:
		    subList.append(n)
		subList.append(X_pred[counter])
                subList.append(0.)
		
                if(X_pred[counter] > 0.5):
		    supList.append(subList)

		counter=counter+1
	
	dataArray=np.array(supList)
	print(np.array(supList).shape)
	
#	dataArray=np.array([supList])
	
	np.savetxt(functionName,dataArray,delimiter=' ')
	

def PlotROCSci(clf,X_test,Y_test):
    X_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    import scikitplot as skplt
    skplt.metrics.plot_confusion_matrix(Y_test, X_pred, normalize=True)
    skplt.metrics.plot_roc(Y_test, y_probas)
    plt.show()


#Only for binary classifier
def CalibrationPlot(X_train,Y_train,X_Test,Y_test):
    import scikitplot as skplt
    X_test=X_Test[:,0:6]
    rf=RandomForest(X_train,Y_train,X_test,Y_test,writeToFile=True)
    gb=GradientBoosting(X_train,Y_train,X_test,Y_test,writeToFile=True)
    dt=DecisionTree(X_train,Y_train,X_test,Y_test)
    lda=LDA(X_train,Y_train,X_test,Y_test)
    nn=NearestNeighbours(X_train,Y_train,X_test,Y_test)
    mlp=MLP(X_train,Y_train,X_test,Y_test)
    adaboost=AdaBoost(X_train,Y_train,X_test,Y_test)

    rf_probas=rf.predict_proba(X_test)
    gb_probas=gb.predict_proba(X_test)
    dt_probas=dt.predict_proba(X_test)
    lda_probas=lda.predict_proba(X_test)
    nn_probas=nn.predict_proba(X_test)
    mlp_probas=mlp.predict_proba(X_test)
    adaboost_probas=rf.predict_proba(X_test)
    
    print(rf_probas.shape)
    
    probas_list = [rf_probas,gb_probas,dt_probas,lda_probas,nn_probas,mlp_probas,adaboost_probas]

    clf_names = ['RandomForest', 'GradientBoosting','DecisionTree', 'LDA','NearestNeighbours','MLP','AdaBoost']
    skplt.metrics.plot_calibration_curve(Y_test, probas_list, clf_names)
    plt.show()



	
def PlotROC(clf,X_test,Y_test):
    import numpy as np
    #===============================================================
    print("========== %s  =========" % whoami())

    model = clf
    y_test = Y_test
    y_predict_proba = model.predict_proba(X_test)

    # Compute ROC curve and ROC AUC for each class
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
	y_test_i = map(lambda x: 1 if x == i else 0, y_test)
	all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
	all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
	fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])

    print("========== Print TPR =============")
    #print(tpr[2])
    # Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = auc(fpr["average"], tpr["average"])


    # Plot average ROC Curve
    plt.figure()
    plt.plot(fpr["average"], tpr["average"],
	     label='Average ROC curve (area = {0:0.2f})'
		   ''.format(roc_auc["average"]),
	     color='deeppink', linestyle=':', linewidth=4)

    # Plot each individual ROC curve
    for i in range(n_classes):
	plt.plot(fpr[i], tpr[i], lw=2,
		 label='ROC curve of class {0} (area = {1:0.2f})'
		 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    

    '''
    print(X_pred.shape)
    supList=[]
    for i in range(X_pred.shape[0]):
        #print(X_pred[i])
        #if(X_pred[i]!=0.):
            #print("Raman")
            subList=[]
            #subList=testDataArr[i,6:9]
            for j in range(6,9):
                subList.append(testDataArr[i,j])
            subList.append(X_pred[i])
            #print(subList)
            supList.append(subList)

    import numpy as np
    filtDataArr=np.array(supList)
    np.savetxt("filteredTestPt.txt",filtDataArr,delimiter=",")
    '''

