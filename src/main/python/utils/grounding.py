from utils.read_write_data import readRawTurkDataFile
from utils.read_write_data import loadEmbeddings
from utils.read_write_data import writeToFile
from utils.read_write_data import writeToFileWithPd
from utils.linearReg import runLR
from tqdm import tqdm
import numpy as np
import math
import sys
import torchtext.vocab as vocab
import torchwordemb

cbow4 = "glove_vectors_syn_ant_sameord_difford.txt"

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




def get_features_y(cwd, turkFile, useOneHot):
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
        #np.random.seed(1)
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
        # # write training data to a separate file. This should happen only once.
        # for eachline in trainingData_indices:
        #     results = [df_raw_turk_data["turker"][eachline],df_raw_turk_data["adjective"][eachline],df_raw_turk_data["mean"][eachline],
        #                df_raw_turk_data["onestdev"][eachline],
        #                df_raw_turk_data["had_negative"][eachline],df_raw_turk_data["logrespdev"][eachline]]
        #
        #     trainingData.append(results )
        #
        # writeToFile(trainingData,cwd, "trainingData.csv")
        #
        #
        #
        #
        # #split the rest into half as dev and test
        # dev_test_indices=np.array_split(rest,2)
        # #print(len(rest))
        #
        #
        # dev_indices=dev_test_indices[0]
        # test_indices=dev_test_indices[1]
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
        # # write test data to a separate file. This should happen only once.
        # test_list=[]
        # for eachline in test_indices:
        #     results = [df_raw_turk_data["turker"][eachline],df_raw_turk_data["adjective"][eachline],df_raw_turk_data["mean"][eachline],
        #                df_raw_turk_data["onestdev"][eachline],
        #                df_raw_turk_data["had_negative"][eachline],df_raw_turk_data["logrespdev"][eachline]]
        #
        #     test_list.append(results )
        #
        #
        #
        # writeToFile(test_list,cwd, "test.csv")
        # sys.exit(1)


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
        with open("trainingData.csv", 'a') as f:
            for rowCounter, eachTurkRow in tqdm(enumerate(trainingData_indices),total=len(trainingData_indices), desc="readV:"):

                #write the training data to a file
                # slice = df_raw_turk_data.iloc[eachTurkRow]
                # #trainingData.append(slice)
                # slice.to_csv(f, sep=',',header=False,index=False,index_label=False)
                # slice = df_raw_turk_data.iloc[eachTurkRow+1]
                # slice.to_csv(f, sep=',', header=False, index=False, index_label=False)
                #
                #
                # writeToFileWithPd(df_raw_turk_data, cwd, "trainingData.csv")


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



        return npfeatures,y, uniq_adj, all_adj




