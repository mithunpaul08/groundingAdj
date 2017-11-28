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
        df_raw_turk_data=readRawTurkDataFile(cwd, turkFile)
        noOfRows=df_raw_turk_data.shape[0]

        #create an numpy array of that range
        allIndex=np.arange(noOfRows)

        #now shuffle it and split
        np.random.shuffle(allIndex)

        splitTurk=np.array_split(allIndex,2)

        #print(df_raw_turk_data["logrespdev"][0])

        trainingData=splitTurk[0]
        rest=splitTurk[1]

        #split the rest into half as dev and test
        dev_test=np.array_split(rest,2)
        dev=dev_test[0]
        test=dev_test[1]

        #print(trainingData.shape)
        #print(dev.shape)
        #print(test.shape)

        #print(trainingData["mean"])

        #For each line in the training part of turk data document:Get the mean and variance from that line in the turk document
        for eachTurkRow in trainingData:
            #give this index to the actual data frame
            mean=df_raw_turk_data["mean"][eachTurkRow]
            variance=df_raw_turk_data["onestdev"][eachTurkRow]
            print("index:"+str(eachTurkRow))
            print("mean"+str(mean))
            sys.exit(1)




    ##################################end of dev phase####################
    except:
        import traceback
        print('generic exception: ' + traceback.format_exc())
        elapsed_time = time.time() - start_time
        print("time taken:" + str(elapsed_time))






