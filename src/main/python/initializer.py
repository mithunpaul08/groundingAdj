from __future__ import division
import scipy
import os
import sys;
import utils;
import csv;
import collections
import numpy as np
import itertools
from utils.read_data import readAdjInterceptFile
from utils.read_data import readRawTurkDataFile
import pickle as pk
from scipy.stats import kendalltau, spearmanr

import time





start_time = time.time()

cwd=os.getcwd()
turkFile="adjectiveData.csv"

if __name__ == "__main__":
    try:
        rawTurkData=readRawTurkDataFile(cwd,turkFile)
        #print(rawTurkData["logrespdev"][0])
        splitTurk=np.array_split(rawTurkData,2)

        trainingData=splitTurk[0]
        rest=splitTurk[1]

        #split the rest into half as dev and test
        dev_test=np.array_split(rest,2)
        dev=dev_test[0]
        test=dev_test[1]

        print(trainingData.shape)
        print(dev.shape)
        print(test.shape)




    ##################################end of dev phase####################
    except:
        import traceback
        print('generic exception: ' + traceback.format_exc())
        elapsed_time = time.time() - start_time
        print("time taken:" + str(elapsed_time))






