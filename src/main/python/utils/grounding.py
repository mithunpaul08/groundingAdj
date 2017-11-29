from utils.read_data import readRawTurkDataFile
from utils.read_data import loadEmbeddings
from utils.linearReg import runLR
from tqdm import tqdm
import numpy as np
import sys
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



