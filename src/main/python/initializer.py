from __future__ import division
import scipy
import os
import sys;
import utils;
import csv;
import collections
import numpy as np
import itertools
from utils.read_write_data import readAdjInterceptFile
from utils.read_write_data import readRawTurkDataFile
from utils.read_write_data import readWithSpace
from utils.squish import calculateRSq
import pickle as pk
from scipy.stats import kendalltau, spearmanr
from utils.linearReg import runLR

import time

from utils.grounding import predict_grounding
from utils.grounding import get_features_dev
from utils.grounding import get_features_training_data
from utils.squish import run_adj_emb
from utils.squish import run_adj_emb_loocv
from sklearn.metrics import r2_score

start_time = time.time()

cwd=os.getcwd()
dev="dev.csv"
training_data="trainingData.csv"
turkInterceptFile="turk_with_intercept.txt"

if __name__ == "__main__":
    try:





        while True:
                print("                      ")
                print("          ******            ")

                print("Welcome to Grounding For Adjectives. Please pick one of the following:")

                print("To run the model with just linear regression Press:1")
                print("To run the model with a dense NN, embeddings through linear regression Press:2")
                print("To run the model with a dense NN, with LOOCV Press:3")
                print("To exit Press:0")


                myInput=input("what is your choice:")

                if(myInput=="2"):

                    #readtraining data
                    uniq_turker={}

                    features, y, adj_lexicon,all_adj,uniq_turker= get_features_training_data (cwd, training_data,False,uniq_turker)








                    adj_lexicon_flipped = dict()
                    #total number of unique adjectives
                    num_adj = len(adj_lexicon)

                    #key=index value=adjective
                    for a, idx in adj_lexicon.items():
                        adj_lexicon_flipped[idx] = a


                    #run with leae one out cross validation
                    #run_adj_emb_loocv(features,y,adj_lexicon,all_adj)

                    #run just with a classic train-dev-test partition
                    trained_model=run_adj_emb(features,y,adj_lexicon,all_adj)

                    print("done training . Going to  read dev data")

                    #read dev data
                    features, y, adj_lexicon,all_adj=   get_features_dev(cwd, dev,False,uniq_turker)
                    print("done reading dev data:")
                    rsquared_value=calculateRSq(y,features,all_adj,trained_model)


                    #calculate rsquared
                    rsquared_value=calculateRSq(y,features,all_adj,trained_model)
                    print("rsquared_value:")
                    print(str(rsquared_value))



                    # adj_intercepts_learned = learned_weights[:num_adj]
                    # #pairing weights with adjectives.
                    # adj_pairs = [(learned_weights[0][i], adj_lexicon_flipped[i]) for i in range(num_adj)]
                    #
                    # sorted_adjs = sorted(adj_pairs, key=lambda x: x[0], reverse=True)
                    #
                    # #print highest 20 intercepts and lowest 20 intercepts
                    # print(sorted_adjs[:20])
                    # print(sorted_adjs[-20:])
                    elapsed_time = time.time() - start_time
                    print("time taken:" + str(elapsed_time/60)+"minutes")

                else:
                    if(myInput=="3"):



                        features, y, adj_lexicon,all_adj= get_features_y(cwd, turkFile,False)
                        adj_lexicon_flipped = dict()
                        #total number of unique adjectives
                        num_adj = len(adj_lexicon)

                        #key=index value=adjective
                        for a, idx in adj_lexicon.items():
                            adj_lexicon_flipped[idx] = a


                        #run with leae one out cross validation
                        run_adj_emb_loocv(features,y,adj_lexicon,all_adj)

                        #run just with a classic train-dev-test partition
                        elapsed_time = time.time() - start_time
                        print("time taken:" + str(elapsed_time/60)+"minutes")
                    else:
                        if(myInput=="0"):
                            print("******Good Bye")
                            break;
                        else:
                            if(myInput=="1"):

                                features, y, adj_lexicon,all_adj= get_features_y(cwd, turkFile,False)
                                print(features.shape)

                                adj_lexicon_flipped = dict()
                                #total number of unique adjectives
                                num_adj = len(adj_lexicon)

                                #key=index value=adjective
                                for a, idx in adj_lexicon.items():
                                    adj_lexicon_flipped[idx] = a

                                #actual linear regression part- how much weight should it assigne to each of 1-hot-adj-vector, mean and variance

                                #will be of size 1x100=98 adj, one mean and variance
                                learned_weights = runLR(features, y)

                                #print(str(learned_weights.shape))
                                #sys.exit(1)
                                #print("NumUniqueAdj: ", num_adj)
                                # Get the weights that correspond to the individual adjs
                                adj_intercepts_learned = learned_weights[:num_adj]
                                #pairing weights with adjectives.
                                adj_pairs = [(learned_weights[0][i], adj_lexicon_flipped[i]) for i in range(num_adj)]

                                #print(adj_pairs[:2])

                                #sorting them by their weight
                                sorted_adjs = sorted(adj_pairs, key=lambda x: x[0], reverse=True)

                                #print highest 20 intercepts and lowest 20 intercepts
                                print(sorted_adjs[:20])
                                print(sorted_adjs[-20:])

    ##################################end of dev phase####################
    except:
        import traceback
        print('generic exception: ' + traceback.format_exc())
        elapsed_time = time.time() - start_time
        print("time taken:" + str(elapsed_time))






