from utils.read_data import readRawTurkDataFile
from utils.read_data import loadEmbeddings
from utils.linearReg import runLR
from tqdm import tqdm
import numpy as np
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

        ##print(df_raw_turk_data["logrespdev"][0])

        trainingData=splitTurk[0]
        rest=splitTurk[1]

        #split the rest into half as dev and test
        dev_test=np.array_split(rest,2)
        dev=dev_test[0]
        test=dev_test[1]

        ##print(trainingData.shape)
        ##print(dev.shape)
        ##print(test.shape)

        ##print(trainingData["mean"])
        y=np.array([])
        features=np.ndarray(shape=(1,302))
        pathDM = cwd + "/data/" + cbow4
        marneffe_data = loadEmbeddings(pathDM)
        print("done reading embeddings. total number of rows/words in this is:")
        print(len(marneffe_data))
        oovCount=0

        #For each line in the training part of turk data document:Get the mean and variance from that line in the turk document
        for eachTurkRow in tqdm(trainingData,total=len(trainingData),desc="turk_data:"):
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            stddev=df_raw_turk_data["onestdev"][eachTurkRow]
            logRespDev=df_raw_turk_data["logrespdev"][eachTurkRow]
            #print("index:"+str(eachTurkRow))
            #print("mean"+str(mean))
            #print("adjective:"+str(adj))

            #print("done reading embeddings. total number of rows/words in this is:")
            #print(len(marneffe_data))
            if(adj not in marneffe_data):
                oovCount=oovCount+1
            else:
                oneoutput=marneffe_data[adj]
                print("shape of oneoutput  is:")
                print((oneoutput.shape))
                withmean=np.append(oneoutput,mean)
                withstd = np.append(withmean, stddev)
                #print(len(withstd))
                #print(logRespDev)
                print("size of withstd is:")
                print((withstd.shape))
                #print("size of y is:")
                #print((y.shape))
                ylabelLocal=np.array([logRespDev])
                featuresLocal = np.asarray(withstd)
                featuresLocal=featuresLocal.transpose()
                print("size of featuresLocal is:")
                print((featuresLocal.shape))
                print("size of ylabelLocal is:")
                print((ylabelLocal.shape))
                print("size of big features is:")
                print((features.shape))

                #print("logrespdev")
                combinedY=np.append(y,ylabelLocal)
                combinedFeatures=np.append(features,featuresLocal,axis=0)
                features=combinedFeatures
                y=combinedY

                print("size of big features is:")
                print((features.shape))
                print("size of big y is:")
                print((y.shape))
                sys.exit(1)


        print("OOV word count is:"+str(oovCount))
        return features, y




def get_features_y(cwd, turkFile, useOneHot):
        df_raw_turk_data=readRawTurkDataFile(cwd, turkFile)
        print(df_raw_turk_data["adjective"][0])

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

        vocab, vec = torchwordemb.load_glove_text("/data/nlp/corpora/glove/6B/glove.6B.300d.txt")
        print(vec.size())
        emb=vec[vocab["apple"]].numpy()
        #print(emb)



        y=np.array([],dtype="float32")
        features = []

        # #for each of the adjective create a one hot vector
        for rowCounter, eachTurkRow in tqdm(enumerate(trainingData),total=len(trainingData), desc="readV:"):

            ########create a one hot vector for adjective
            # give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]

            #get the index of the adjective
            adjIndex=uniq_adj[adj]
            #print("adjIndex:"+str(adjIndex))
            #print("uniq_adj_count:"+str(uniq_adj_count))

            embV=[]
            if(useOneHot):
                #####create a one hot vector for all adjectives
                # one_hot_adj=np.zeros(uniq_adj_count)
                one_hot_adj = [0] * uniq_adj_count
                # print(one_hot_adj)
                # print("one hot shape:"+str((one_hot_adj.shape)))
                one_hot_adj[adjIndex] = 1
                # print(one_hot_adj)
                #todo : extend/append this new vector
                embV=one_hot_adj

            else:
                #pick the corresponding embedding from glove
                emb = vec[vocab[adj]].numpy()
                embV=emb

            ################to create a one hot vector for turker data also
            #get the id number of of the turker
            turkerId=df_raw_turk_data["turker"][eachTurkRow]
            turkerIndex=uniq_turker[turkerId]
            #print("turkerIndex:"+str(turkerIndex))

            #create a one hot vector for all turkers
            one_hotT=[0]*(uniq_turker_count)
            #print(one_hotT)
            #print("one one_hotT shape:"+str((one_hotT.shape)))
            one_hotT[turkerIndex]=1
            #print(one_hotT)


            ################get the mean and variance for this row and attach to this one hot
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            stddev=df_raw_turk_data["onestdev"][eachTurkRow]
            logRespDev=df_raw_turk_data["respdev"][eachTurkRow]
            #print("index:"+str(eachTurkRow))
            #print("mean"+str(mean))
            #print("adjective:"+str(adj))

            #############combine adj-1-hot to mean , variance and turker-one-hot

            localFeatures=[]
            #print("one hot shape:"+str(len(one_hot_adj)))
            #print(" localFeatures shape:"+str(len(localFeatures)))
            localFeatures.extend(embV)

            print(" mean :"+str(type(mean.item())))
            print(" localFeatures shape:"+str(len(localFeatures)))
            localFeatures.append(mean.item())
            #print(localFeatures)
            #print(" localFeatures shape:"+str(len(localFeatures)))
            #print(" stddev :"+str((stddev)))
            localFeatures.append(stddev)
            #localFeatures.extend(one_hotT)
            #print(" localFeatures shape:"+str(len(localFeatures)))


            #print("size of adj_mean_stddev_turk is:")
            #print((adj_mean_stddev_turk.shape))



            ############feed this combined vector as a feature vector to the linear regression
            # print(len(withstd))
            #print(logRespDev)

            #print("size of y is:")
            #print((y.shape))

            ylabelLocal=np.array([logRespDev], dtype="float32")
            #featuresLocal = np.array([adj_mean_stddev_turk])
            #featuresLocal=featuresLocal.transpose()
            #print("size of featuresLocal is:")
            #print((featuresLocal.shape))
            #print("size of ylabelLocal is:")
            #print((ylabelLocal.shape))
            features.append(localFeatures)

            #print("logrespdev")
            combinedY=np.append(y,ylabelLocal)
            #combinedFeatures=np.append(features,featuresLocal,axis=0)
            #features=combinedFeatures
            y=combinedY

        print("size of big features 1is:")
        print(len(features))
        #print("size of big y is:")
        #print((y.shape))
        npfeatures=np.asarray(features, dtype="float32")
        print("size of big features 2is:")
        print((npfeatures.shape))
        return npfeatures,y, uniq_adj




