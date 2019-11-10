from helpers_V2 import *
from sys import argv
modelName=argv[1]
loopVal=int(argv[2])
numOfSamples=int(argv[3])
TuneIt(modelName,loopVal,numOfSamples)
