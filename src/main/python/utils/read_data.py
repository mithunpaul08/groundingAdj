from csv import DictReader
import os
import sys
import csv
import pandas as pd

def readFile(cwd, inputFile):

    path = cwd+"/data/"
    data =pd.read_csv(path  + inputFile,sep=',',header=None,names=['adj','intercept'])
    data['adj'] = data['adj'].map(lambda x: x.lstrip('(').rstrip(')'))
    data['intercept'] = data['intercept'].map(lambda x: x.lstrip('(').rstrip(')'))

    return data;

