from csv import DictReader
import os
import sys
import csv
import pandas as pd

def readAdjInterceptFile(cwd, inputFile):

    path = cwd+"/data/"
    data =pd.read_csv(path  + inputFile,sep=',',header=None,names=['adj','intercept'])
    data['adj'] = data['adj'].map(lambda x: x.lstrip('(').rstrip(')'))
    data['intercept'] = data['intercept'].map(lambda x: x.lstrip('(').rstrip(')'))

    return data;

def readRawTurkDataFile(cwd, inputFile):

    path = cwd+"/data/"
    data =pd.read_csv(path  + inputFile,sep=',')

    #print(data["logrespdev"][0])
    #shuffle the data to make sure we have mixed it evenly
    #data= data.sample(frac=1).reset_index(drop=True)
    #print(data["logrespdev"][0])

    return data;

