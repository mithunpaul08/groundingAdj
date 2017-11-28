from utils.read_data import readRawTurkDataFile
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

        #For each line in the training part of turk data document:Get the mean and variance from that line in the turk document
        for eachTurkRow in trainingData:
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            variance=df_raw_turk_data["onestdev"][eachTurkRow]
            print("index:"+str(eachTurkRow))
            print("mean"+str(mean))
            print("adjective:"+str(adj))
            marneffe_data= utils.read_data.readFile(cwd, paragram1)


