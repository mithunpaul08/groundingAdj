import sys
from utils.read_data import readRawTurkDataFile
from utils.read_data import readFile
import numpy as np

def predict_grounding(cwd,turkFile):
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
        cbow4 = "glove_vectors_syn_ant_sameord_difford.txt"

        #For each line in the training part of turk data document:Get the mean and variance from that line in the turk document
        for eachTurkRow in trainingData:
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            variance=df_raw_turk_data["onestdev"][eachTurkRow]
            print("index:"+str(eachTurkRow))
            print("mean"+str(mean))
            print("adjective:"+str(adj))
            marneffe_data= readFile(cwd, cbow4)



def with_one_hot_adj(cwd,turkFile):
        df_raw_turk_data=readRawTurkDataFile(cwd, turkFile)
        print(df_raw_turk_data["adjective"][0])

        #create a hash table to store unique adj
        uniq_adj={}
        counter=1

        #create a total list of unique adj in this collection
        for a in df_raw_turk_data["adjective"]:

            if(a) not in uniq_adj:
                #if its not there already add it as the latest element
                uniq_adj[a]=counter
                counter=counter+1


   
        uniq_turker={}
        turk_counter=1
        #create a total list of unique turkers in this collection
        for b in df_raw_turk_data["turker"]:
            if(b) not in uniq_turker:
                #if its not there already add it as the latest element
                uniq_turker[b]=turk_counter
                turk_counter=turk_counter+1

        uniq_adj_count=len(uniq_adj)
        uniq_turker_count=len(uniq_turker)


        print("total number of unique adjectives is "+str(len(uniq_adj)))
        print("total number of unique turkers is "+str(len(uniq_turker)))




        #Split data in to train-dev-test

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




        # #for each of the adjective create a one hot vector
        for eachTurkRow in trainingData:
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]

            #get the index of the adjective
            adjIndex=uniq_adj[adj]
            print("adjIndex:"+str(adjIndex))
            print("uniq_adj_count:"+str(uniq_adj_count))

            #create a one hot vector for all adjectives
            one_hot=np.zeros(uniq_adj_count)
            print(one_hot)
            print("one hot shape:"+str((one_hot.shape)))
            one_hot[adjIndex]=1

            print(one_hot)



            sys.exit(1)
