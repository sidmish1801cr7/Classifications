import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
import sys
import time

from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F

from MyDict import *

import sys
def whoami():
     return sys._getframe(1).f_code.co_name


class EnSemble():
     model=KNeighborsClassifier() 
     '''
     X_train=np.array([])
     Y_train=np.array([])
     X_test=np.array([])
     Y_test=np.array([])
     '''
     def __init__(self): #,X_tr,Y_tr,X_te,Y_te):
         '''
         X_train=X_tr
         Y_train=Y_tr
         X_test=X_te
         Y_test=Y_te
         '''
         print("Inside Constructor")
        
         print("========== %s Classifier =========" % whoami())
         from sklearn.ensemble import VotingClassifier
         dt=DecisionTree.model#.fit(X_train,Y_train)
         nn=KNeighbors(3).model#.fit(X_train,Y_traini)
         lda=LinearDiscriminantAnalysis.model

         #estimators=[('gb', gb), ('dt', dt),  ('mlp', mlp),('rf',rf), ('nn',nn),('lda',lda)]
         estimators=[ ('dt', dt),('nn',nn),('lda',lda)]
         model = VotingClassifier(estimators, voting='soft')

         #return clf


class KNeighbors():
    num_neighbours=1
    model=KNeighborsClassifier() 

    def __init__(self,num=1):
        num_neighbours=num
        print("Generating model for num_neighbours : "+str(num_neighbours))
        model=KNeighborsClassifier(n_neighbors=num_neighbours)
    

    

    #def Model(self):
     #   return model
        


class DecisionTree():
    estimators=1
    model=tree.DecisionTreeClassifier()

	
class RandomForest():
    num_estimators=1
    model=RandomForestClassifier()
    def __init__(self,num):
        num_estimators=num
        model=RandomForestClassifier(n_estimators=num_estimators)

class LDA():
    model=LinearDiscriminantAnalysis()



