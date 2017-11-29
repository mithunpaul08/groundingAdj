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

from utils.grounding import predict_grounding



start_time = time.time()

cwd=os.getcwd()
turkFile="adjectiveData.csv"

if __name__ == "__main__":
    try:

        features,y =predict_grounding(cwd,turkFile)
        print("size of features is:")
        print((features.shape))
        print("size of y is:")
        print((y.shape))

    ##################################end of dev phase####################
    except:
        import traceback
        print('generic exception: ' + traceback.format_exc())
        elapsed_time = time.time() - start_time
        print("time taken:" + str(elapsed_time))






