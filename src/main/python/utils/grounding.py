import math
import numpy as np
import sys
import torchtext.vocab as vocab
import torchwordemb
from tqdm import tqdm

from utils.linearReg import runLR
from utils.read_write_data import loadEmbeddings
from utils.read_write_data import readRawTurkDataFile
from utils.read_write_data import writeCsvToFile
from utils.read_write_data import writeToFileWithHeader
from utils.read_write_data import writeToFileWithPd

cbow4 = "glove_vectors_syn_ant_sameord_difford.txt"
random_seed=1
useRandomSeed=False

def predict_grounding(cwd,turkFile):
        df_raw_turk_data=readRawTurkDataFile(cwd, turkFile)
        noOfRows=df_raw_turk_data.shape[0]

        #create an numpy array of that range
        allIndex=np.arange(noOfRows)

        #now shuffle it and split
        np.random.shuffle(allIndex)

        splitTurk=np.array_split(allIndex,2)

        ###print(df_raw_turk_data["logrespdev"][0])

        trainingData=splitTurk[0]
        rest=splitTurk[1]

        #split the rest into half as dev and test
        dev_test=np.array_split(rest,2)
        dev=dev_test[0]
        test=dev_test[1]

        ###print(trainingData.shape)
        ###print(dev.shape)
        ###print(test.shape)

        ###print(trainingData["mean"])
        y=np.array([])
        features=np.ndarray(shape=(1,302))
        pathDM = cwd + "/data/" + cbow4
        marneffe_data = loadEmbeddings(pathDM)
        ##print("done reading embeddings. total number of rows/words in this is:")
        ##print(len(marneffe_data))
        oovCount=0

        #For each line in the training part of turk data document:Get the mean and variance from that line in the turk document
        for eachTurkRow in tqdm(trainingData,total=len(trainingData),desc="turk_data:"):
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            stddev=df_raw_turk_data["onestdev"][eachTurkRow]
            logRespDev=df_raw_turk_data["logrespdev"][eachTurkRow]
            ##print("index:"+str(eachTurkRow))
            ##print("mean"+str(mean))
            ##print("adjective:"+str(adj))

            ##print("done reading embeddings. total number of rows/words in this is:")
            ##print(len(marneffe_data))
            if(adj not in marneffe_data):
                oovCount=oovCount+1
            else:
                oneoutput=marneffe_data[adj]
                ##print("shape of oneoutput  is:")
                ##print((oneoutput.shape))
                withmean=np.append(oneoutput,mean)
                withstd = np.append(withmean, stddev)
                ##print(len(withstd))
                ##print(logRespDev)
                ##print("size of withstd is:")
                ##print((withstd.shape))
                #print("size of y is:")
                #print((y.shape))
                ylabelLocal=np.array([logRespDev])
                featuresLocal = np.asarray(withstd)
                featuresLocal=featuresLocal.transpose()
                # print("size of featuresLocal is:")
                # print((featuresLocal.shape))
                # print("size of ylabelLocal is:")
                # print((ylabelLocal.shape))
                # print("size of big features is:")
                # print((features.shape))

                #print("logrespdev")
                combinedY=np.append(y,ylabelLocal)
                combinedFeatures=np.append(features,featuresLocal,axis=0)
                features=combinedFeatures
                y=combinedY

                # print("size of big features is:")
                # print((features.shape))
                # print("size of big y is:")
                # print((y.shape))
                sys.exit(1)


        ##print("OOV word count is:"+str(oovCount))
        return features, y



#take all the data in the given file.
def get_features_dev(cwd, turkFile, useOneHot,uniq_turker,addTurkerOneHot):
        df_raw_turk_data=readRawTurkDataFile(cwd, turkFile)

        #create a hash table to store unique adj
        uniq_adj={}
        counter=0

        #create a total list of unique adj in this collection
        for a in df_raw_turk_data["adjective"]:
            if(a) not in uniq_adj:
                #if its not there already add it as the latest element
                uniq_adj[a]=counter
                counter=counter+1



        turk_counter=0
        #create a total list of unique turkers in this collection
        for b in df_raw_turk_data["turker"]:
            if(b) not in uniq_turker:
                #if its not there already add it as the latest element
                uniq_turker[b]=turk_counter
                turk_counter=turk_counter+1

        uniq_adj_count=len(uniq_adj)
        uniq_turker_count=len(uniq_turker)


        ##print("total number of unique adjectives is "+str(len(uniq_adj)))
        ##print("total number of unique turkers is "+str(len(uniq_turker)))




        #Split data in to train-dev-test
        noOfRows=df_raw_turk_data.shape[0]
        #print("noOfRows")
        #print(noOfRows)


        #create an numpy array of that range
        allIndex=np.arange(noOfRows)

        #now shuffle it and split
        #np.random.seed(1)
        #np.random.shuffle(allIndex)

        #take 80% of the total data as training data- rest as testing
        #eighty=math.ceil(noOfRows*80/100)
        #twenty_index=math.ceil(noOfRows*80/100)
        #print("eighty")
        #print(eighty)

        #eighty= number of rows
        #trainingData_indices=allIndex[:eighty]
        #rest=allIndex[eighty:]






        y=np.array([],dtype="float32")
        features = []


        #list of all adjectives in the training data, including repeats
        all_adj=[]



        data_indices=np.arange(noOfRows)

        # #for each of the adjective create a one hot vector
        for rowCounter, eachTurkRow in tqdm(enumerate(data_indices),total=noOfRows, desc="readData:"):

            ########create a one hot vector for adjective
            # give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            all_adj.append(adj)

            #get the index of the adjective
            adjIndex=uniq_adj[adj]
            ##print("adjIndex:"+str(adjIndex))
            ##print("uniq_adj_count:"+str(uniq_adj_count))

            embV=[]
            if(useOneHot):
                #####create a one hot vector for all adjectives
                # one_hot_adj=np.zeros(uniq_adj_count)
                one_hot_adj = [0] * uniq_adj_count
                # #print(one_hot_adj)
                # #print("one hot shape:"+str((one_hot_adj.shape)))
                one_hot_adj[adjIndex] = 1
                # #print(one_hot_adj)
                #todo : extend/append this new vector
                embV=one_hot_adj

            else:
                #pick the corresponding embedding from glove
                #emb = vec[vocab[adj]].numpy()
                embV=embV
                #embV=emb

            ################to create a one hot vector for turker data also
            #get the id number of of the turker
            turkerId=df_raw_turk_data["turker"][eachTurkRow]
            turkerIndex=uniq_turker[turkerId]
            ##print("turkerIndex:"+str(turkerIndex))

            #create a one hot vector for all turkers
            one_hotT=[0]*(uniq_turker_count)
            ##print(one_hotT)
            ##print("one one_hotT shape:"+str((one_hotT.shape)))
            one_hotT[turkerIndex]=1
            ##print(one_hotT)


            ################get the mean and variance for this row and attach to this one hot
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            stddev=df_raw_turk_data["onestdev"][eachTurkRow]
            logRespDev=df_raw_turk_data["logrespdev"][eachTurkRow]
            ##print("index:"+str(eachTurkRow))
            ##print("mean"+str(mean))
            ##print("adjective:"+str(adj))

            #############combine adj-1-hot to mean , variance and turker-one-hot

            localFeatures=[]
            ##print("one hot shape:"+str(len(one_hot_adj)))
            ##print(" localFeatures shape:"+str(len(localFeatures)))
            #localFeatures.extend(embV)

            ##print(" mean :"+str(type(mean.item())))
            ##print(" localFeatures shape:"+str(len(localFeatures)))
            localFeatures.append(mean.item())
            ##print(localFeatures)
            ##print(" localFeatures shape:"+str(len(localFeatures)))
            ##print(" stddev :"+str((stddev)))
            localFeatures.append(stddev)
            if(addTurkerOneHot):
                localFeatures.extend(one_hotT)
            ##print(" localFeatures shape:"+str(len(localFeatures)))


            ##print("size of adj_mean_stddev_turk is:")
            ##print((adj_mean_stddev_turk.shape))



            ############feed this combined vector as a feature vector to the linear regression
            # #print(len(withstd))
            ##print(logRespDev)

            ##print("size of y is:")
            ##print((y.shape))

            ylabelLocal=np.array([logRespDev], dtype="float32")
            #featuresLocal = np.array([adj_mean_stddev_turk])
            #featuresLocal=featuresLocal.transpose()
            ##print("size of featuresLocal is:")
            ##print((featuresLocal.shape))
            ##print("size of ylabelLocal is:")
            ##print((ylabelLocal.shape))
            features.append(localFeatures)

            ##print("logrespdev")
            combinedY=np.append(y,ylabelLocal)
            #combinedFeatures=np.append(features,featuresLocal,axis=0)
            #features=combinedFeatures
            y=combinedY





        #print("size of big features 1is:")
        # #print(len(features))

        npfeatures=np.asarray(features, dtype="float32")
        # #print("size of big features 2is:")
        # #print((npfeatures.shape))
        # #print("size of big y is:")
        # #print((y.shape))
        #
        # #print("size of uniq_adj is:")
        # #print(len(uniq_adj))
        #
        # #print("size of all_adj is:")
        # #print(len(all_adj))
        # total=len(all_adj)
        #
        # #print(all_adj[0])
        # #print(all_adj[total-1])



        return npfeatures,y, uniq_adj, all_adj

def get_features_labels_from_data(cwd, turkFile, useAdjOneHot, uniq_turker, addTurkerOneHot):
        df_raw_turk_data=readRawTurkDataFile(cwd, turkFile)

        #create a hash table to store unique adj
        uniq_adj={}
        counter=0

        #create a dictionary  of unique adj in this collection
        for a in df_raw_turk_data["adjective"]:
            if(a) not in uniq_adj:
                #if its not there already add it as the latest element
                uniq_adj[a]=counter
                counter=counter+1

        uniq_adj_list=[]
        # create a list  of unique adj in this collection
        for a in uniq_adj.keys():
                uniq_adj_list.append(a)



        turk_counter=0
        #create a total list of unique turkers in this collection
        for b in df_raw_turk_data["turker"]:
            if(b) not in uniq_turker:
                #if its not there already add it as the latest element
                uniq_turker[b]=turk_counter
                turk_counter=turk_counter+1

        uniq_adj_count=len(uniq_adj)
        uniq_turker_count=len(uniq_turker)


        ##print("total number of unique adjectives is "+str(len(uniq_adj)))
        ##print("total number of unique turkers is "+str(len(uniq_turker)))




        #Split data in to train-dev-test
        noOfRows=df_raw_turk_data.shape[0]
        #print("noOfRows")
        #print(noOfRows)


        #create an numpy array of that range
        allIndex=np.arange(noOfRows)

        #now shuffle it and split
        #np.random.seed(1)
        #np.random.shuffle(allIndex)

        #take 80% of the total data as training data- rest as testing
        #eighty=math.ceil(noOfRows*80/100)
        #twenty_index=math.ceil(noOfRows*80/100)
        #print("eighty")
        #print(eighty)

        #eighty= number of rows
        #trainingData_indices=allIndex[:eighty]
        #rest=allIndex[eighty:]






        y=np.array([],dtype="float32")
        features = []


        #list of all adjectives in the training data, including repeats
        all_adj=[]



        data_indices=np.arange(noOfRows)

        # #for each of the adjective create a one hot vector
        for rowCounter, eachTurkRow in tqdm(enumerate(data_indices),total=noOfRows, desc="readData:"):

            ########create a one hot vector for adjective
            # give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            all_adj.append(adj)

            #get the index of the adjective
            adjIndex=uniq_adj[adj]
            ##print("adjIndex:"+str(adjIndex))
            ##print("uniq_adj_count:"+str(uniq_adj_count))

            embV=[]
            if(useAdjOneHot):
                #####create a one hot vector for all adjectives
                # one_hot_adj=np.zeros(uniq_adj_count)
                one_hot_adj = [0] * uniq_adj_count
                # #print(one_hot_adj)
                # #print("one hot shape:"+str((one_hot_adj.shape)))
                one_hot_adj[adjIndex] = 1
                # #print(one_hot_adj)
                #todo : extend/append this new vector
                embV=one_hot_adj

            else:
                #pick the corresponding embedding from glove
                #emb = vec[vocab[adj]].numpy()
                embV=embV
                #embV=emb

            ################to create a one hot vector for turker data also
            #get the id number of of the turker
            turkerId=df_raw_turk_data["turker"][eachTurkRow]
            turkerIndex=uniq_turker[turkerId]
            ##print("turkerIndex:"+str(turkerIndex))

            #create a one hot vector for all turkers
            one_hotT=[0]*(uniq_turker_count)
            ##print(one_hotT)
            ##print("one one_hotT shape:"+str((one_hotT.shape)))
            one_hotT[turkerIndex]=1
            ##print(one_hotT)


            ################get the mean and variance for this row and attach to this one hot
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            stddev=df_raw_turk_data["onestdev"][eachTurkRow]
            logRespDev=df_raw_turk_data["logrespdev"][eachTurkRow]
            ##print("index:"+str(eachTurkRow))
            ##print("mean"+str(mean))
            ##print("adjective:"+str(adj))

            #############combine adj-1-hot to mean , variance and turker-one-hot

            localFeatures=[]
            ##print("one hot shape:"+str(len(one_hot_adj)))
            ##print(" localFeatures shape:"+str(len(localFeatures)))
            #localFeatures.extend(embV)

            ##print(" mean :"+str(type(mean.item())))
            ##print(" localFeatures shape:"+str(len(localFeatures)))
            localFeatures.append(mean.item())
            ##print(localFeatures)
            ##print(" localFeatures shape:"+str(len(localFeatures)))
            ##print(" stddev :"+str((stddev)))
            localFeatures.append(stddev)
            if(addTurkerOneHot):
                localFeatures.extend(one_hotT)
            ##print(" localFeatures shape:"+str(len(localFeatures)))


            ##print("size of adj_mean_stddev_turk is:")
            ##print((adj_mean_stddev_turk.shape))



            ############feed this combined vector as a feature vector to the linear regression
            # #print(len(withstd))
            ##print(logRespDev)

            ##print("size of y is:")
            ##print((y.shape))

            ylabelLocal=np.array([logRespDev], dtype="float32")
            #featuresLocal = np.array([adj_mean_stddev_turk])
            #featuresLocal=featuresLocal.transpose()
            ##print("size of featuresLocal is:")
            ##print((featuresLocal.shape))
            ##print("size of ylabelLocal is:")
            ##print((ylabelLocal.shape))
            features.append(localFeatures)

            ##print("logrespdev")
            combinedY=np.append(y,ylabelLocal)
            #combinedFeatures=np.append(features,featuresLocal,axis=0)
            #features=combinedFeatures
            y=combinedY




        # #print("size of big features 1is:")
        # #print(len(features))

        npfeatures=np.asarray(features, dtype="float32")
        # #print("size of big features 2is:")
        # #print((npfeatures.shape))
        # #print("size of big y is:")
        # #print((y.shape))
        #
        # #print("size of uniq_adj is:")
        # #print(len(uniq_adj))
        #
        # #print("size of all_adj is:")
        # #print(len(all_adj))
        # total=len(all_adj)
        #
        # #print(all_adj[0])
        # #print(all_adj[total-1])



        return npfeatures,y, uniq_adj, all_adj,uniq_turker,uniq_adj_list


'''this is created on jan 22nd 2018. By now we have the grounding working for squished to 1 architecture.
Turker is removed. Next aim is to sort the entire data based on 98 adjectives, split it into 80-10-10 based on
adjectives. So that 10 data points in TEST partiition will be unseen by the code '''
def split_data_based_on_adj(cwd, turkFile, useOneHot, uniq_turker):
    df_raw_turk_data = readRawTurkDataFile(cwd, turkFile)

    #sort the entire data based on adjectives.
    sorted_df_raw_turk_data=df_raw_turk_data.sort_values("adjective")

    print(sorted_df_raw_turk_data)
    split_adj_data_write_to_file(sorted_df_raw_turk_data, cwd)

    sys.exit(1)

    # create a hash table to store unique adj
    uniq_adj = {}
    counter = 0

    # create a total list of unique adj in this collection
    for a in df_raw_turk_data["adjective"]:
        if (a) not in uniq_adj:
            # if its not there already add it as the latest element
            uniq_adj[a] = counter
            counter = counter + 1

    turk_counter = 0
    # create a total list of unique turkers in this collection
    for b in df_raw_turk_data["turker"]:
        if (b) not in uniq_turker:
            # if its not there already add it as the latest element
            uniq_turker[b] = turk_counter
            turk_counter = turk_counter + 1

    uniq_adj_count = len(uniq_adj)
    uniq_turker_count = len(uniq_turker)

    ##print("total number of unique adjectives is "+str(len(uniq_adj)))
    ##print("total number of unique turkers is "+str(len(uniq_turker)))




    # Split data in to train-dev-test
    noOfRows = df_raw_turk_data.shape[0]
    # print("noOfRows")
    # print(noOfRows)


    # create an numpy array of that range
    allIndex = np.arange(noOfRows)

    # now shuffle it and split
    # np.random.seed(1)
    # np.random.shuffle(allIndex)

    # take 80% of the total data as training data- rest as testing
    # eighty=math.ceil(noOfRows*80/100)
    # twenty_index=math.ceil(noOfRows*80/100)
    # print("eighty")
    # print(eighty)

    # eighty= number of rows
    # trainingData_indices=allIndex[:eighty]
    # rest=allIndex[eighty:]






    y = np.array([], dtype="float32")
    features = []

    # list of all adjectives in the training data, including repeats
    all_adj = []

    data_indices = np.arange(noOfRows)

    # #for each of the adjective create a one hot vector
    for rowCounter, eachTurkRow in tqdm(enumerate(data_indices), total=noOfRows, desc="readData:"):

        ########create a one hot vector for adjective
        # give this index to the actual data frame
        adj = df_raw_turk_data["adjective"][eachTurkRow]
        all_adj.append(adj)

        # get the index of the adjective
        adjIndex = uniq_adj[adj]
        ##print("adjIndex:"+str(adjIndex))
        ##print("uniq_adj_count:"+str(uniq_adj_count))

        embV = []
        if (useOneHot):
            #####create a one hot vector for all adjectives
            # one_hot_adj=np.zeros(uniq_adj_count)
            one_hot_adj = [0] * uniq_adj_count
            # #print(one_hot_adj)
            # #print("one hot shape:"+str((one_hot_adj.shape)))
            one_hot_adj[adjIndex] = 1
            # #print(one_hot_adj)
            # todo : extend/append this new vector
            embV = one_hot_adj

        else:
            # pick the corresponding embedding from glove
            # emb = vec[vocab[adj]].numpy()
            embV = embV
            # embV=emb

        ################to create a one hot vector for turker data also
        # get the id number of of the turker
        turkerId = df_raw_turk_data["turker"][eachTurkRow]
        turkerIndex = uniq_turker[turkerId]
        ##print("turkerIndex:"+str(turkerIndex))

        # create a one hot vector for all turkers
        one_hotT = [0] * (uniq_turker_count)
        ##print(one_hotT)
        ##print("one one_hotT shape:"+str((one_hotT.shape)))
        one_hotT[turkerIndex] = 1
        ##print(one_hotT)


        ################get the mean and variance for this row and attach to this one hot
        # give this index to the actual data frame
        adj = df_raw_turk_data["adjective"][eachTurkRow]
        mean = df_raw_turk_data["mean"][eachTurkRow]
        stddev = df_raw_turk_data["onestdev"][eachTurkRow]
        logRespDev = df_raw_turk_data["logrespdev"][eachTurkRow]
        ##print("index:"+str(eachTurkRow))
        ##print("mean"+str(mean))
        ##print("adjective:"+str(adj))

        #############combine adj-1-hot to mean , variance and turker-one-hot

        localFeatures = []
        ##print("one hot shape:"+str(len(one_hot_adj)))
        ##print(" localFeatures shape:"+str(len(localFeatures)))
        # localFeatures.extend(embV)

        ##print(" mean :"+str(type(mean.item())))
        ##print(" localFeatures shape:"+str(len(localFeatures)))
        localFeatures.append(mean.item())
        ##print(localFeatures)
        ##print(" localFeatures shape:"+str(len(localFeatures)))
        ##print(" stddev :"+str((stddev)))
        localFeatures.append(stddev)
        # localFeatures.extend(one_hotT)
        ##print(" localFeatures shape:"+str(len(localFeatures)))


        ##print("size of adj_mean_stddev_turk is:")
        ##print((adj_mean_stddev_turk.shape))



        ############feed this combined vector as a feature vector to the linear regression
        # #print(len(withstd))
        ##print(logRespDev)

        ##print("size of y is:")
        ##print((y.shape))

        ylabelLocal = np.array([logRespDev], dtype="float32")
        # featuresLocal = np.array([adj_mean_stddev_turk])
        # featuresLocal=featuresLocal.transpose()
        ##print("size of featuresLocal is:")
        ##print((featuresLocal.shape))
        ##print("size of ylabelLocal is:")
        ##print((ylabelLocal.shape))
        features.append(localFeatures)

        ##print("logrespdev")
        combinedY = np.append(y, ylabelLocal)
        # combinedFeatures=np.append(features,featuresLocal,axis=0)
        # features=combinedFeatures
        y = combinedY

    # #print("size of big features 1is:")
    # #print(len(features))

    npfeatures = np.asarray(features, dtype="float32")
    # #print("size of big features 2is:")
    # #print((npfeatures.shape))
    # #print("size of big y is:")
    # #print((y.shape))
    #
    # #print("size of uniq_adj is:")
    # #print(len(uniq_adj))
    #
    # #print("size of all_adj is:")
    # #print(len(all_adj))
    # total=len(all_adj)
    #
    # #print(all_adj[0])
    # #print(all_adj[total-1])



    return npfeatures, y, uniq_adj, all_adj, uniq_turker

'''split the given data set into training, dev and test partitions'''
def split_entire_data(cwd, turkFile, useOneHot):
        df_raw_turk_data=readRawTurkDataFile(cwd, turkFile)

        #create a hash table to store unique adj
        uniq_adj={}
        counter=0

        #create a total list of unique adj in this collection
        for a in df_raw_turk_data["adjective"]:
            if(a) not in uniq_adj:
                #if its not there already add it as the latest element
                uniq_adj[a]=counter
                counter=counter+1



        uniq_turker={}
        turk_counter=0
        #create a total list of unique turkers in this collection
        for b in df_raw_turk_data["turker"]:
            if(b) not in uniq_turker:
                #if its not there already add it as the latest element
                uniq_turker[b]=turk_counter
                turk_counter=turk_counter+1

        uniq_adj_count=len(uniq_adj)
        uniq_turker_count=len(uniq_turker)


        ##print("total number of unique adjectives is "+str(len(uniq_adj)))
        ##print("total number of unique turkers is "+str(len(uniq_turker)))




        #Split data in to train-dev-test
        noOfRows=df_raw_turk_data.shape[0]
        #print("noOfRows")
        #print(noOfRows)


        #create an numpy array of that range
        allIndex=np.arange(noOfRows)

        #now shuffle it and split
        if(useRandomSeed):
            np.random.seed(random_seed)

        np.random.shuffle(allIndex)

        #take 80% of the total data as training data- rest as testing
        eighty=math.ceil(noOfRows*80/100)
        #twenty_index=math.ceil(noOfRows*80/100)
        #print("eighty")
        #print(eighty)

        #eighty= number of rows
        trainingData_indices=allIndex[:eighty]
        rest=allIndex[eighty:]



        trainingData=[]
        # write training data to a separate file. This should happen only once.
        # for eachline in trainingData_indices:
        #     results = [df_raw_turk_data["turker"][eachline],df_raw_turk_data["adjective"][eachline],df_raw_turk_data["mean"][eachline],
        #                df_raw_turk_data["onestdev"][eachline],
        #                df_raw_turk_data["had_negative"][eachline],df_raw_turk_data["logrespdev"][eachline]]
        #
        #     trainingData.append(results )
        #
        # writeToFileWithHeader(trainingData,cwd, "trainingData_with_random.csv")


        #
        #
        #
        #
        #split the rest into half as dev and test
        dev_test_indices=np.array_split(rest,2)
        #print(len(rest))
        #
        #
        dev_indices=dev_test_indices[0]
        test_indices=dev_test_indices[1]
        #
        #
        # # write dev data to a separate file. This should happen only once.
        # dev_list=[]
        # for eachline in dev_indices:
        #     results = [df_raw_turk_data["turker"][eachline],df_raw_turk_data["adjective"][eachline],df_raw_turk_data["mean"][eachline],
        #                df_raw_turk_data["onestdev"][eachline],
        #                df_raw_turk_data["had_negative"][eachline],df_raw_turk_data["logrespdev"][eachline]]
        #
        #     dev_list.append(results)
        # writeToFile(dev_list,cwd, "dev.csv")
        #
        # write test data to a separate file. This should happen only once.
        test_list=[]
        for eachline in test_indices:
            results = [df_raw_turk_data["turker"][eachline],df_raw_turk_data["adjective"][eachline],df_raw_turk_data["mean"][eachline],
                       df_raw_turk_data["onestdev"][eachline],
                       df_raw_turk_data["had_negative"][eachline],df_raw_turk_data["logrespdev"][eachline]]

            test_list.append(results )



        writeToFileWithHeader(test_list,cwd, "test_rand_seed1.csv")
        sys.exit(1)


        y=np.array([],dtype="float32")
        features = []

        trainingData = []

        #list of all adjectives in the training data, including repeats
        all_adj=[]

        # with open("trainingData.csv", 'w') as f:
        #     df_raw_turk_data.iloc[1].to_csv(f,header=False)
        #     df_raw_turk_data.iloc[2].to_csv(f, sep=',', header=False, index=False, index_label=False)
        #
        # sys.exit(1)


        # #for each of the adjective create a one hot vector
        for rowCounter, eachTurkRow in tqdm(enumerate(trainingData_indices),total=len(trainingData_indices), desc="readTrngData:"):

            ########create a one hot vector for adjective
            # give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            all_adj.append(adj)

            #get the index of the adjective
            adjIndex=uniq_adj[adj]
            ##print("adjIndex:"+str(adjIndex))
            ##print("uniq_adj_count:"+str(uniq_adj_count))

            embV=[]
            if(useOneHot):
                #####create a one hot vector for all adjectives
                # one_hot_adj=np.zeros(uniq_adj_count)
                one_hot_adj = [0] * uniq_adj_count
                # #print(one_hot_adj)
                # #print("one hot shape:"+str((one_hot_adj.shape)))
                one_hot_adj[adjIndex] = 1
                # #print(one_hot_adj)
                #todo : extend/append this new vector
                embV=one_hot_adj

            else:
                #pick the corresponding embedding from glove
                #emb = vec[vocab[adj]].numpy()
                embV=embV
                #embV=emb

            ################to create a one hot vector for turker data also
            #get the id number of of the turker
            turkerId=df_raw_turk_data["turker"][eachTurkRow]
            turkerIndex=uniq_turker[turkerId]
            ##print("turkerIndex:"+str(turkerIndex))

            #create a one hot vector for all turkers
            one_hotT=[0]*(uniq_turker_count)
            ##print(one_hotT)
            ##print("one one_hotT shape:"+str((one_hotT.shape)))
            one_hotT[turkerIndex]=1
            ##print(one_hotT)


            ################get the mean and variance for this row and attach to this one hot
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            stddev=df_raw_turk_data["onestdev"][eachTurkRow]
            logRespDev=df_raw_turk_data["logrespdev"][eachTurkRow]
            ##print("index:"+str(eachTurkRow))
            ##print("mean"+str(mean))
            ##print("adjective:"+str(adj))

            #############combine adj-1-hot to mean , variance and turker-one-hot

            localFeatures=[]
            ##print("one hot shape:"+str(len(one_hot_adj)))
            ##print(" localFeatures shape:"+str(len(localFeatures)))
            #localFeatures.extend(embV)

            ##print(" mean :"+str(type(mean.item())))
            ##print(" localFeatures shape:"+str(len(localFeatures)))
            localFeatures.append(mean.item())
            ##print(localFeatures)
            ##print(" localFeatures shape:"+str(len(localFeatures)))
            ##print(" stddev :"+str((stddev)))
            localFeatures.append(stddev)
            localFeatures.extend(one_hotT)
            ##print(" localFeatures shape:"+str(len(localFeatures)))


            ##print("size of adj_mean_stddev_turk is:")
            ##print((adj_mean_stddev_turk.shape))



            ############feed this combined vector as a feature vector to the linear regression
            # #print(len(withstd))
            ##print(logRespDev)

            ##print("size of y is:")
            ##print((y.shape))

            ylabelLocal=np.array([logRespDev], dtype="float32")
            #featuresLocal = np.array([adj_mean_stddev_turk])
            #featuresLocal=featuresLocal.transpose()
            ##print("size of featuresLocal is:")
            ##print((featuresLocal.shape))
            ##print("size of ylabelLocal is:")
            ##print((ylabelLocal.shape))
            features.append(localFeatures)

            ##print("logrespdev")
            combinedY=np.append(y,ylabelLocal)
            #combinedFeatures=np.append(features,featuresLocal,axis=0)
            #features=combinedFeatures
            y=combinedY




        # #print("size of big features 1is:")
        # #print(len(features))

        npfeatures=np.asarray(features, dtype="float32")
        # #print("size of big features 2is:")
        # #print((npfeatures.shape))
        # #print("size of big y is:")
        # #print((y.shape))
        #
        # #print("size of uniq_adj is:")
        # #print(len(uniq_adj))
        #
        # #print("size of all_adj is:")
        # #print(len(all_adj))
        # total=len(all_adj)
        #
        # #print(all_adj[0])
        # #print(all_adj[total-1])



        return npfeatures,y, uniq_adj, all_adj,uniq_turker


def split_adj_data_write_to_file(df_raw_turk_data, cwd):
    #df_raw_turk_data = readRawTurkDataFile(cwd, turkFile)

    # # create a hash table to store unique adj
    uniq_adj = {}
    counter = 0

    # create a total list of unique adj in this collection
    for a in df_raw_turk_data["adjective"]:
        if (a) not in uniq_adj:
            # if its not there already add it as the latest element
            uniq_adj[a] = counter
            counter = counter + 1

    print("total number of unique adjectives is "+str(len(uniq_adj)))




    # get the total number of unique adjectives
    noOfRows = len(uniq_adj.items())

    # noOfRows_data = len(df_raw_turk_data.items())

    # print("noOfRows")
    # print(noOfRows)


    # create an numpy array of that range
    allIndex = np.arange(noOfRows)

    # now shuffle it and split
    # np.random.seed(1)
    #np.random.shuffle(allIndex)

    # take 80% of the total data as training data- rest as testing
    eighty = math.ceil(noOfRows * 80 / 100)
    ten= math.ceil(noOfRows * 10 / 100)

    # twenty_index=math.ceil(noOfRows*80/100)
    # print("eighty")
    # print(eighty)


    uniq_adj_list=[]
    for k,v in uniq_adj.items():
        uniq_adj_list.append(k)


    # eighty= number of rows
    trainingData_adj = uniq_adj_list[:eighty]
    dev_adj = uniq_adj_list[eighty:(eighty+ten)]
    test_adj = uniq_adj_list[(eighty+ten):(eighty+ten+ten)]

    #
    # print("trainingData_adj")
    # print(len(trainingData_adj))
    # print("dev_adj")
    # print((dev_adj))
    # print("test_adj")
    # print((test_adj))


    #go through each of the index in training data.

    trainingData = []
    devData=[]
    testData=[]

    # write training data to a separate file. This should happen only once.
    for index, eachline in df_raw_turk_data.iterrows():
        thisadj=eachline['adjective']

        results = [eachline["turker"],eachline["adjective"],eachline["mean"],
                   eachline["onestdev"],
                   eachline["had_negative"],eachline["logrespdev"]]

        if(thisadj in trainingData_adj):
            trainingData.append(results)
        else:
            if(thisadj in dev_adj):
                devData.append(results)
            else:
                if(thisadj in test_adj):
                    testData.append(results)


    writeToFileWithHeader(trainingData,cwd, "trainingData_adj.csv")
    writeToFileWithHeader(devData,cwd, "dev_adj.csv")
    writeToFileWithHeader(testData,cwd, "test_adj.csv")


    sys.exit(1)


