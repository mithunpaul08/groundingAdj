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
from utils.linearReg import runLR

import time

from utils.grounding import predict_grounding
from utils.grounding import get_features_y_one_hot


start_time = time.time()

cwd=os.getcwd()
turkFile="adjectiveData.csv"

if __name__ == "__main__":
    try:

        features, y, adj_lexicon= get_features_y_one_hot(cwd, turkFile)

        adj_lexicon_flipped = dict()
        num_adj = len(adj_lexicon)
        for a, idx in adj_lexicon.items():
            adj_lexicon_flipped[idx] = a
        learned_weights = runLR(features, y)
        print("NumUniqueAdj: ", num_adj)
        adj_intercepts = learned_weights[:num_adj]
        adj_pairs = [(learned_weights[0][i], adj_lexicon_flipped[i]) for i in range(num_adj)]
        print(adj_pairs[:2])
        sorted_adjs = sorted(adj_pairs, key=lambda x: x[0], reverse=True)
        print(sorted_adjs[:20])
        print(sorted_adjs[-20:])


        # features,y =predict_grounding(cwd,turkFile)
        # print("size of features is:")
        # print((features.shape))
        # print("size of y is:")
        # print((y.shape))

    ##################################end of dev phase####################
    except:
        import traceback
        print('generic exception: ' + traceback.format_exc())
        elapsed_time = time.time() - start_time
        print("time taken:" + str(elapsed_time))






