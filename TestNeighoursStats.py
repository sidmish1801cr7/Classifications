from TestReadModel_Plot import *
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    scoreList=[]
    numNeighList=[3,5,8,10]
    for numNeigh in numNeighList:
        score=Process(argv[1],argv[2],argv[3],argv[4],numNeigh)
        scoreList.append(score)

    print("Score List : "+score)

