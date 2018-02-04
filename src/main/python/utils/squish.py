import pickle as pk
import sys
import torch
import math
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab
import torchwordemb
import numpy as np
import os
from itertools import accumulate, chain, repeat, tee
from utils.linearReg import convert_variable
from utils.read_write_data import readRawTurkDataFile
from sklearn.metrics import r2_score
from torch.autograd import Variable
from tqdm import tqdm

from utils.grounding import get_features_dev
torch.manual_seed(1)

#hidden_layers=[30,1]
# no_of_hidden_layers=3
dense1_size=1
#dense2_size=1
# dense3_size=1

noOfFoldsCV=30
noOfEpochs=135
lr=1e-5
#lr=1e-2

rsq_file="rsq_file.txt"
rsq_file_nfcv="rsq_file_nfcv.txt"
rsq_file_nfcv_avrg="rsq_file_nfcv_avrg.txt"


class AdjEmb(nn.Module):
    #the constructor. Pass whatever you need to
    def __init__(self,turkCount,addTurkerOneHot):
        super(AdjEmb,self).__init__()


        # get teh glove vectors
        print("going to load glove for per adj.")

        # get the glove embeddings for this adjective
        #self.vocab, self.vec = torchwordemb.load_glove_text("/data/nlp/corpora/glove/6B/glove.6B.300d.txt")

        #load a subset of glove which contains embeddings for the adjectives we have

        cwd=os.getcwd()
        path = cwd+"/data/"
        self.vocab, self.vec = torchwordemb.load_glove_text(path+"glove_our_adj.csv")

        #emb=self.vec[self.vocab["intense"]]
        #print(emb)


        self.noOfTurkers=turkCount



        # df_raw_turk_data=readRawTurkDataFile(cwd,"glove_our_adj")
        # print(df_raw_turk_data)
        # print(df_raw_turk_data["mithun"])
        #sys.exit(1)

        # get teh glove vectors
        print("just loaded glove for per adj. going to load glove for entire embeddings.")

        print(".self.vec.shape[0]")
        print(self.vec.shape[0])
        print(".self.vec.shape[1]")
        print(self.vec.shape[1])
        self.embeddings = nn.Embedding(self.vec.shape[0], self.vec.shape[1])
        self.embeddings.weight.data.copy_(self.vec)

        #dont update embeddings
        self.embeddings.weight.requires_grad=True


        #the size of the last layer will be the last entry in dense3_size
        #dense3_size=hidden_layers[len(hidden_layers)-1]


        # the layer where you squish the 300 embeddings to a dense layer of whatever size.
        # if the list hidden_layers[] is empty, then it will be directly squished to tbd

        # i.e it takes embeddings as input and returns a dense layer of size 10
        # note: this is also known as the weight vector to be used in an affine

        self.linear1 = nn.Linear(self.vec.size(1), dense1_size)
        #self.linear2 = torch.nn.Linear(dense1_size, dense2_size)
        # self.linear3 = torch.nn.Linear(dense2_size, dense3_size)


        #dynamically add the hidden layers
        # for index,layer in enumerate(hidden_layers):
        #     #first squish alone is from size of embeddings
        #     if(index==0):
        #         layername="linear"+str(index)
        #         self.layername = nn.Linear(self.vec.size(1), hidden_layers[index])
        #     else:
        #         #second squish onwards dynamically add
        #         if((index+1) < len(hidden_layers)):
        #             layername="linear"+str(index)
        #             self.layername = torch.nn.Linear(hidden_layers[index], hidden_layers[index+1])




        #
        # print("dense3_size:")
        # print(dense3_size)

        #the last step: whatever the output of previous layer was concatenate it with the mu and sigma and one-hot vector for turker
        if(addTurkerOneHot):
            self.fc = torch.nn.Linear(dense1_size+turkCount+2, 1)
            print("found addTurkerOneHot=true")
        else:
            #use this when you dont have one hot for turkers
            self.fc = torch.nn.Linear(dense1_size+2, 1)



        print("done loading all gloves")

        #glove = vocab.GloVe(name='6B', dim=300)
        #the linear regression code which maps hidden layer to intercept value must come here


    #init was where you where just defining what embeddings meant. Here we actually use it
    def forward(self, adj, feats):

        #get the corresponding  embeddings of the adjective
        #emb_adj=self.embeddings(adj)




        #print(self.vec.size())
        #print("adj:")
        #print(adj)
        emb=self.vec[self.vocab[adj]]#.numpy()
        #embT =torch.from_numpy(emb)
        embV=Variable(emb,requires_grad=False)

        #
        out=F.tanh(self.linear1(embV))
        #out=F.tanh(self.linear2(out))
        #out=F.tanh(self.linear3(out))


        # #dynamically add the hidden layers
        # for index,layer in enumerate(hidden_layers):
        #     layername="linear"+str(index)
        #     #first squish alone is from size of embeddings
        #     if(index==0):
        #         out=F.tanh(self.layername(embV))
        #     else:
        #         #second squish onwards dynamically add
        #
        #         if((index+1) < len(hidden_layers)):
        #             out=F.tanh(self.layername(out))






        feature_squished = (torch.cat((feats, out)))

        retValue=(self.fc(feature_squished))
        return retValue



#########################actual feed forward neural network related class and code ends here
#for each word  append its correesponding index, convert it to a long tensor, then to a variable and return
def getIndex(w, to_ix):
    #idxs=[0, 1, 2, 3, 4]
    #print(w)
    idxs = [to_ix[w]]
    #print(idxs)
    #print("just printed indices")
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)




#take the adjectives and give it all an id
def convert_adj_index(listOfAdj):

    adj_Indices={}
    for eachAdj in tqdm(listOfAdj,total=len(listOfAdj),desc="adj_train:"):
        # print("eachSent:")
        # print(eachSent)
        if eachAdj not in adj_Indices:
                    adj_Indices[eachAdj] = len(adj_Indices)
    return adj_Indices




    file_Name2 = "adj_Indices.pkl"
    # open the file for writing
    fileObject2 = open(file_Name2,'wb')
    pk.dump(adj_Indices, fileObject2)



def convert_scalar_to_variable(features):

    x2 =torch.from_numpy(np.array([features]))

    return Variable(x2)

def convert_to_variable(features):

    x2 =torch.from_numpy(features)

    return Variable(x2)



#the actual trainign code. Basically create an object of the class above
def do_training(features, allY, list_Adj, all_adj):
    #take the list of adjectives and give it all an index
    adj_index=convert_adj_index(list_Adj)

    print("got inside do_training. going to Load embeddings:")

    #there are 193 unique turkers
    model=AdjEmb(193)

    #rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)

    #run through each epoch, feed forward, calculate loss, back propagate

    #no point having epoch if you are not back propagating
    #for epoch in tqdm(range(no_of_epochs),total=no_of_epochs,desc="squishing:"):

    #things needed for the linear regression phase
    featureShape=features.shape

    params_to_update = filter(lambda p: p.requires_grad==True, model.parameters())
    rms = optim.RMSprop(params_to_update,lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    loss_fn = nn.MSELoss(size_average=True)

    allIndex = np.arange(len(features))


    #keep one out

    #train on the rest, test on this one, add it to the
    #print(w2v.vec[ w2v.words["large"] ] )

    for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):
        #for each word in the list of adjectives


        pred_y_total=[]
        y_total=[]
        adj_10_emb={}

        #shuffle for each epoch
        np.random.shuffle(allIndex)

        for eachRow in tqdm(allIndex, total=len(features), desc="each_adj:"):

            model.zero_grad()

            feature=features[eachRow]
            y = allY[eachRow]
            each_adj = all_adj[eachRow]

            featureV= convert_to_variable(feature)
            pred_y = model(each_adj, featureV)





            adj_10_emb[each_adj]=pred_y


            #the complete linear regression code- only thing is features here will include the squished_emb
            # Reset gradients

            true_variable_y = convert_scalar_to_variable(y)
            y_total.append(y)

            #rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
            #print("batch_x")
            #print(batch_x)

            #multiply weight with input vector
            # affine=fc(batch_x)
            #
            # #this is the actual prediction of the intercept
            # pred_y=affine.data.cpu().numpy()
            pred_y_total.append(pred_y.data.cpu().numpy()[0])




            loss = loss_fn(pred_y, true_variable_y)




            # Backward pass
            loss.backward()



            # optimizer.step()
            # adam.step()
            rms.step()

        # print("zip")
        # print(list(zip(y_total,pred_y_total)))
        # # print("y_total:")
        # # print(y_total)
        # # print("pred_y_total:")
        # # print(pred_y_total)
        rsquared_value=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')
        # print("rsquared_value")
        # print(rsquared_value)
        # print("loss:")
        # print(loss)



    print("done training the model. Going to  write it to disk")


    #the model is trained by now-store it to disk
    file_Name5 = "squish.pkl"
    # open the file for writing
    fileObject5 = open(file_Name5,'wb')
    pk.dump(model, fileObject5)

   #  learned_weights = fc.weight.data
   #  #return(learned_weights.cpu().numpy())
   #
   #
   # #save the weights to disk
   #  file_Name1 = "learned_weights.pkl"
   #  # open the file for writing
   #  fileObject1 = open(file_Name1,'wb')
   #  pk.dump(learned_weights.cpu().numpy(), fileObject1)

    return model



    # print("loss")
    #
    # #print(loss)
    # #
    # #
    # # #todo: return the entire new 98x10 hashtable to regression code
    # # print(adj_10_emb)
    # # sys.exit(1)
    # #
    # # print('Loss: after all epochs'+str((loss.data)))
    # #
    # #print("allY value:")
    # #print(len(y_total))
    # #print("predicted allY value")
    # #print(len(pred_y_total))
    # rsquared_value=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')
    #
    #
    # print("rsquared_value:")
    # print(str(rsquared_value))
    #learned_weights = model.affine.weight.data
    #return(learned_weights.cpu().numpy())

    # #rsquared_value2= rsquared(allY, pred_y)
    # print("rsquared_value2:")
    # print(str(rsquared_value2))

    # print(fc.weight.data.view(-1))

'''train on training data, print its rsquared against the same training data, then test with dev, print its rsquared. Do this at each epoch.
This is all done for tuning purposes'''
def  train_dev_print_rsq(dev,features, allY, list_Adj, all_adj,uniq_turker,addTurkerOneHot):
    #take the list of adjectives and give it all an index
    adj_index=convert_adj_index(list_Adj)

    print("got inside do_training. going to Load embeddings:")

    #there are 193 unique turkers
    model=AdjEmb(193,addTurkerOneHot)


    params_to_update = filter(lambda p: p.requires_grad==True, model.parameters())
    rms = optim.RMSprop(params_to_update,lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    loss_fn = nn.MSELoss(size_average=True)

    allIndex = np.arange(len(features))

    cwd = os.getcwd()

    #empty out the existing file
    with open(cwd + "/outputs/" + rsq_file, "w+")as rsq_values:
        rsq_values.write("Epoch \t Train \t\t Dev \n")
    rsq_values.close()

    #append the rest of the values
    with open(cwd+"/outputs/" +rsq_file,"a")as rsq_values:
        for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):
            pred_y_total=[]
            y_total=[]
            adj_10_emb={}

            #shuffle for each epoch
            np.random.shuffle(allIndex)

            for eachRow in tqdm(allIndex, total=len(features), desc="each_adj:"):

                model.zero_grad()

                feature=features[eachRow]
                y = allY[eachRow]
                each_adj = all_adj[eachRow]

                featureV= convert_to_variable(feature)
                pred_y = model(each_adj, featureV)


                adj_10_emb[each_adj]=pred_y

                #the complete linear regression code- only thing is features here will include the squished_emb
                # Reset gradients

                true_variable_y = convert_scalar_to_variable(y)
                y_total.append(y)

                #rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
                #print("batch_x")
                #print(batch_x)

                #multiply weight with input vector
                # affine=fc(batch_x)
                #
                # #this is the actual prediction of the intercept
                # pred_y=affine.data.cpu().numpy()
                pred_y_total.append(pred_y.data.cpu().numpy()[0])




                loss_training = loss_fn(pred_y, true_variable_y)





                # Backward pass
                loss_training.backward()



                # optimizer.step()
                # adam.step()
                rms.step()

            # print("zip")
            # print(list(zip(y_total,pred_y_total)))
            # # print("y_total:")
            # # print(y_total)
            # # print("pred_y_total:")
            # # print(pred_y_total)




            rsquared_value_training=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')
            # print("rsquared_value_training: "+str(rsquared_value_training))
            rsq_values.write(str(epoch)+"\t"+str(rsquared_value_training)+"\t")
            # print("loss_training:")
            # print(loss_training)






            tuneOnDev(model,dev,cwd, uniq_turker,rsq_values,rsquared_value_training,loss_training,addTurkerOneHot,epoch)
            # Print weights
            learned_weights = model.fc.weight.data
            print("\tlearned weights:" + str(learned_weights.cpu().numpy()))





    print("done training the model. Going to  write it to disk")


    #the model is trained by now-store it to disk
    file_Name5 = "squish.pkl"
    # open the file for writing
    fileObject5 = open(file_Name5,'wb')
    pk.dump(model, fileObject5)

   #  learned_weights = fc.weight.data
   #  #return(learned_weights.cpu().numpy())
   #
   #
   # #save the weights to disk
   #  file_Name1 = "learned_weights.pkl"
   #  # open the file for writing
   #  fileObject1 = open(file_Name1,'wb')
   #  pk.dump(learned_weights.cpu().numpy(), fileObject1)

    return model



    # print("loss_training")
    #
    # #print(loss_training)
    # #
    # #
    # # #todo: return the entire new 98x10 hashtable to regression code
    # # print(adj_10_emb)
    # # sys.exit(1)
    # #
    # # print('Loss: after all epochs'+str((loss_training.data)))
    # #
    # #print("allY value:")
    # #print(len(y_total))
    # #print("predicted allY value")
    # #print(len(pred_y_total))
    # rsquared_value_training=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')
    #
    #
    # print("rsquared_value_training:")
    # print(str(rsquared_value_training))
    #learned_weights = model.affine.weight.data
    #return(learned_weights.cpu().numpy())

    # #rsquared_value2= rsquared(allY, pred_y)
    # print("rsquared_value2:")
    # print(str(rsquared_value2))

    # print(fc.weight.data.view(-1))



'''  create feed forward NN model, but using loocv for cross validation'''
def run_loocv_on_turk_data(features, allY, uniq_adj, all_adj,addTurkerOneHot):


    #take the list of adjectives and give it all an index
    adj_index=convert_adj_index(uniq_adj)

    print("got inside do_training. going to call model:")

    model=AdjEmb(193,addTurkerOneHot)

    #rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)

    #run through each epoch, feed forward, calculate loss, back propagate


    params_to_update = filter(lambda p: p.requires_grad==True, model.parameters())
    rms = optim.RMSprop(params_to_update,lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    loss_fn = nn.MSELoss(size_average=True)

    allIndex = np.arange(len(features))




    pred_y_total = []
    y_total = []
    adj_10_emb = {}

    rsq_total=[]


    # for each element in the training data, keep that one out, and train on the rest
    for eachElement in tqdm(allIndex,total=len(allIndex), desc="n-fold-CV:"):

        # create a list of all the indices except the one you are keeping out
        allIndex_loocv=[x for x,i in enumerate(allIndex) if i!=eachElement]


        # print("eachElement:")
        # print(eachElement)

        feature = features[eachElement]
        # print("feature of held out one:")
        # print(feature)

        # print("len(trainingData):")
        # print(len(allIndex_loocv))
        # print("the value that was left out was")
        # print(allIndex[eachElement])

        #train on the rest, test on this one left out, add it to the

        for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):

            #for each word in the list of adjectives
            model.zero_grad()



            #shuffle for each epoch
            np.random.shuffle(allIndex_loocv)

            '''for each row in the training data, predict y value for itself, and then back
            propagate the loss'''
            for eachRow in tqdm(allIndex_loocv, total=len(features), desc="each_adj:"):
                # print("eachRow:")
                # print(eachRow)

                #using shuffling
                feature=features[eachRow]

                y = allY[eachRow]
                each_adj = all_adj[eachRow]

                featureV= convert_to_variable(feature)
                pred_y = model(each_adj, featureV)

                adj_10_emb[each_adj]=pred_y
                batch_y = convert_scalar_to_variable(y)

                loss = loss_fn(pred_y, batch_y)

                # Backward pass
                loss.backward()

                rms.step()





        #for loocv use the trained model to predict on the left over value
        feature_loo = features[eachElement]
        featureV_loo= convert_to_variable(feature_loo)
        #print(feature)
        y = allY[eachElement]
        each_adj = all_adj[eachElement]
        pred_y = model(each_adj, featureV_loo)
        #adj_10_emb[each_adj] = pred_y
        batch_y = convert_scalar_to_variable(y)
        y_total.append(y)
        #for each of the entry in training data, predict and store it in a bigger table
        pred_y_total.append(pred_y.data.cpu().numpy())

        # print(y)
        # print(each_adj)
        # print("pred_Y;")
        # print(pred_y)

        # the LOOCV ends here do this for each element as "THE LEAVE ONE OUT" the training data


        #print loss at the end of every element left out

        #print(adj_10_emb)
        # print('Loss: after all epochs'+str((loss.data)))
        print("allY value length (must be 2648):")
        print(len(y_total))
        print("predicted allY value length (must be 2648):")
        print(len(pred_y_total))
        print("loss")
        print(loss)

    print("done with all training data")



    print("allY value length (must be 2648):")
    print(len(y_total))
    print("predicted allY value length (must be 2648):")
    print(len(pred_y_total))


    rsquared_value=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')


    print("rsquared_value:")
    print(str(rsquared_value))

    rsq_total.append(rsquared_value)

    sys.exit(1)
    #learned_weights = model.affine.weight.data
    #return(learned_weights.cpu().numpy())

    # #rsquared_value2= rsquared(allY, pred_y)
    # print("rsquared_value2:")
    # print(str(rsquared_value2))

    # print(fc.weight.data.view(-1))

'''from: http://wordaligned.org/articles/slicing-a-list-evenly-with-python'''
def chunk(xs, n):
    assert n > 0
    L = len(xs)
    s, r = divmod(L, n)
    widths = chain(repeat(s+1, r), repeat(s, n-r))
    offsets = accumulate(chain((0,), widths))
    b, e = tee(offsets)
    next(e)
    return [xs[s] for s in map(slice, b, e)]


'''  create feed forward NN model, but using 100 data points (around 33 folds) for cross validation'''
def run_nfoldCV_on_turk_data(features, allY, uniq_adj, all_adj,addTurkerOneHot,useEarlyStopping):


    allIndex = np.arange(len(features))



    #split it into folds. n=number of folds. almost even sized.
    n=noOfFoldsCV
    split_data=chunk(allIndex,n)
    chunkIndices = np.arange(len(split_data))


    rsq_total=[]
    cwd=os.getcwd()

    # write rsq to disk
    with open(cwd + "/outputs/" + rsq_file_nfcv, "w+")as nfcv:
        #empty out the existing file before loop does append
        nfcv.write("Chunk \t RSQ\n")
        nfcv.close()


    with open(cwd + "/outputs/" + rsq_file_nfcv, "a")as nfcv:
        # for each chunk in the training data, keep that one out, and train on the rest
        # append the rest of the values
        for eachChunkIndex in tqdm(chunkIndices,total=len(chunkIndices), desc="n-fold-CV:"):

            model = AdjEmb(193, addTurkerOneHot)

            params_to_update = filter(lambda p: p.requires_grad == True, model.parameters())
            rms = optim.RMSprop(params_to_update, lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
            loss_fn = nn.MSELoss(size_average=True)

            # create a  list of all the indices of chunks except the chunk you are keeping out
            allIndices_chunks=[]
            for i in chunkIndices:
                if i!=eachChunkIndex:
                    allIndices_chunks.append(i)

            training_data=[]

            #for each of these chunks, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the training data
            for eachChunk in allIndices_chunks:
                for eachElement in split_data[eachChunk]:
                    training_data.append(eachElement)

            test_data=[]
            #for the left out chunk, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the test data
            for eachElement in split_data[eachChunkIndex]:
                    test_data.append(eachElement)


            rsq_max_estop=0.000
            rsq_previous_estop=0.000
            patienceCounter=0;
            patience_max=20;


            #run n epochs on the left over training data
            for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):

                '''adding early-stopping and patience'''
                if(useEarlyStopping):

                    # split the training data further into training and dev
                    len_training_estop = len(training_data)
                    indices_tr_estop = np.arange(len_training_estop)
                    eighty_estop=math.ceil(len_training_estop*80/100)
                    trainingData_estop=indices_tr_estop[:eighty_estop]
                    dev_estop=indices_tr_estop[eighty_estop:]
                    training_data = trainingData_estop

                    #debug statements
                    print("len_training_estop:")
                    print(len_training_estop)
                    print("(trainingData_estop):")
                    print((trainingData_estop))



                #shuffle before each epoch
                np.random.shuffle(training_data)

                print("(training_data):")
                print((training_data))



                '''for each row in the training data, predict y value for itself, and then back
                propagate the loss'''
                for eachRow in tqdm(training_data, total=len(features), desc="each_adj:"):


                    #every time you feed forward, make sure the gradients are emptied out. From pytorch documentation
                    model.zero_grad()

                    feature=features[eachRow]

                    y = allY[eachRow]
                    each_adj = all_adj[eachRow]


                    featureV= convert_to_variable(feature)
                    pred_y = model(each_adj, featureV)


                    batch_y = convert_scalar_to_variable(y)

                    loss = loss_fn(pred_y, batch_y)

                    # Backward pass
                    loss.backward()

                    rms.step()


                '''if using early stopping: For each epoch,
                train on training_data and test on dev.
                Calculate rsq and Store the rsq value if more than previous'''

                if (useEarlyStopping):

                    pred_y_total_dev_data = []
                    y_total_dev_data = []

                    # for each element in the dev data, calculate its predicted value, and append it to predy_total
                    for test_data_index in dev_estop:
                        this_feature = features[test_data_index]
                        featureV_loo = convert_to_variable(this_feature)
                        y = allY[test_data_index]
                        each_adj = all_adj[test_data_index]
                        pred_y = model(each_adj, featureV_loo)
                        y_total_dev_data.append(y)
                        pred_y_total_dev_data.append(pred_y.data.cpu().numpy())

                    # calculate the rsquared value for entire dev_estop
                    rsquared_value_estop = r2_score(y_total_dev_data, pred_y_total_dev_data, sample_weight=None,
                                              multioutput='uniform_average')
                    print("\n")
                    print("rsquared_value_estop:" + str(rsquared_value_estop))
                    print("\n")

                    #in the first epoch all the values are initialized to the current value
                    if(epoch==1):
                        rsq_max_estop = rsquared_value_estop
                        rsq_previous_estop = rsquared_value_estop

                    #2nd epoch onwards keep track of the maximum rsq value so far
                    else:
                        if(rsquared_value_estop>rsq_max_estop):
                            rsq_max_estop = rsquared_value_estop

                    #everytime the current rsquared value is less than the previous value, increase patience count
                    if (rsquared_value_estop < rsq_previous_estop):
                        patienceCounter=patienceCounter+1

                    print("epoch:"+str(epoch)+"rsq_max_estop:"+str(rsq_max_estop))
                    print("rsq_previous_estop"+str(rsq_previous_estop) +"rsquared_value_estop:"+rsquared_value_estop)

                    rsq_previous_estop = rsquared_value_estop

                    sys.exit(1)

                    if(patienceCounter>patience_max):
                        print("losing my patience. Crossed 20. Exiting")
                        print("rsq_max_estop:"+str(rsq_max_estop))
                        sys.exit(1)





            #at the end of all epochs take the trained model that was trained on the 29 epochs
            #and use the trained model to predict on the values in the left over chunk


            pred_y_total_test_data = []
            y_total_test_data = []


            #for each element in the test data, calculate its predicted value, and append it to predy_total
            for test_data_index in test_data:

                this_feature = features[test_data_index]
                featureV_loo= convert_to_variable(this_feature)
                y = allY[test_data_index]
                each_adj = all_adj[test_data_index]
                pred_y = model(each_adj, featureV_loo)
                y_total_test_data.append(y)
                pred_y_total_test_data.append(pred_y.data.cpu().numpy())


            #calculate the rsquared value for each chunk
            rsquared_value=r2_score(y_total_test_data, pred_y_total_test_data, sample_weight=None, multioutput='uniform_average')
            print("\n")
            print("rsquared_value:"+str(rsquared_value))
            print("\n")
            nfcv.write(str(eachChunkIndex) + "\t" + str(rsquared_value) + "\n")
            rsq_total.append(rsquared_value)


    #  After all chunks are done, calculate the average of each element in the list of predicted rsquared values.
    # There should be 30 such values,
    # each corresponding to one chunk being held out



    rsq_cumulative=0;

    for eachRsq in rsq_total:
        rsq_cumulative=rsq_cumulative+eachRsq


    rsq_average=rsq_cumulative/(len(rsq_total))

    print("rsq_average:")
    print(str(rsq_average))

    # empty out the existing file
    with open(cwd + "/outputs/" + rsq_file_nfcv_avrg, "w+")as rsq_values_avg:
        rsq_values_avg.write("rsq_average: \t "+str(rsq_average))
    rsq_values_avg.close()


    sys.exit(1)


'''  create feed forward NN model, but using loocv for cross validation'''
def run_loocv_per_adj(features, allY, uniq_adj, all_adj,addTurkerOneHot,uniq_adj_list):




    print("got inside run_loocv_per_adj_. going to call model:")

    model=AdjEmb(193,addTurkerOneHot)

    params_to_update = filter(lambda p: p.requires_grad==True, model.parameters())
    rms = optim.RMSprop(params_to_update,lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    loss_fn = nn.MSELoss(size_average=True)

    allIndex = np.arange(len(uniq_adj))




    pred_y_total = []
    y_total = []
    adj_10_emb = {}


    # for each element in the list of adjectives keep that one out, and train on the rest
    # instead of shuffling a list of adjectives, just create an index of all adjectives and shuffle that. easier to do
    for index,eachElement in tqdm(enumerate(allIndex),total=len(allIndex), desc="eachTrngData:"):

        # create a list of all the indices except the one you are keeping out
        allIndex_loocv=[x for x,i in enumerate(allIndex) if i!=eachElement]


        print("eachElement:")
        print(eachElement)

        feature = features[eachElement]
        print("feature of held out one:")
        print(feature)

        print("len(trainingData):")
        print(len(allIndex_loocv))
        print("the value that was left out was")
        print(allIndex[eachElement])
        leftOutIndex=allIndex[eachElement]

        print(("the adjective that was left out was"))
        leftOutAdj=uniq_adj_list[leftOutIndex]
        print(leftOutAdj)

        if (index == 5):
            sys.exit(1)

        continue



        #train on the rest
        for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):

            #for each word in the list of adjectives
            model.zero_grad()


            #shuffle before each epoch
            np.random.shuffle(allIndex_loocv)

            '''for each row in the training data, check if its adjective is part of the held out one. If yes, add to test set.
             Else predict y value for itself, and then back
            propagate the loss'''
            for eachRow in tqdm(allIndex_loocv, total=len(features), desc="each_adj:"):
                # print("eachRow:")
                # print(eachRow)

                #using shuffling
                feature=features[eachRow]

                y = allY[eachRow]
                each_adj = all_adj[eachRow]

                featureV= convert_to_variable(feature)
                pred_y = model(each_adj, featureV)

                adj_10_emb[each_adj]=pred_y
                batch_y = convert_scalar_to_variable(y)

                loss = loss_fn(pred_y, batch_y)

                # Backward pass
                loss.backward()

                rms.step()





        #for loocv use the trained model to predict on the left over value
        feature_loo = features[eachElement]
        featureV_loo= convert_to_variable(feature_loo)
        #print(feature)
        y = allY[eachElement]
        each_adj = all_adj[eachElement]
        pred_y = model(each_adj, featureV_loo)
        #adj_10_emb[each_adj] = pred_y
        batch_y = convert_scalar_to_variable(y)
        y_total.append(y)
        #for each of the entry in training data, predict and store it in a bigger table
        pred_y_total.append(pred_y.data.cpu().numpy())

        # print(y)
        # print(each_adj)
        # print("pred_Y;")
        # print(pred_y)

        # the LOOCV ends here do this for each element as "THE LEAVE ONE OUT" the training data


        #print loss at the end of every element left out

        #print(adj_10_emb)
        # print('Loss: after all epochs'+str((loss.data)))
        print("allY value length (must be 2648):")
        print(len(y_total))
        print("predicted allY value length (must be 2648):")
        print(len(pred_y_total))
        print("loss")
        print(loss)

    print("done with all training data")
   #  #the model is trained by now-store it to disk
   #  file_Name5 = "squish.pkl"
   #  # open the file for writing
   #  fileObject5 = open(file_Name5,'wb')
   #  pk.dump(model, fileObject5)
   #
   #  learned_weights = fc.weight.data
   #  #return(learned_weights.cpu().numpy())
   #
   #
   # #save the weights to disk
   #  file_Name1 = "learned_weights.pkl"
   #  # open the file for writing
   #  fileObject1 = open(file_Name1,'wb')
   #  pk.dump(learned_weights.cpu().numpy(), fileObject1)



    #print("loss")
    #print(loss)
    # print(adj_10_emb)
    # print('Loss: after all epochs'+str((loss.data)))
    print("allY value length (must be 2648):")
    print(len(y_total))
    print("predicted allY value length (must be 2648):")
    print(len(pred_y_total))


    rsquared_value=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')


    print("rsquared_value:")
    print(str(rsquared_value))

    sys.exit(1)
    #learned_weights = model.affine.weight.data
    #return(learned_weights.cpu().numpy())

    # #rsquared_value2= rsquared(allY, pred_y)
    # print("rsquared_value2:")
    # print(str(rsquared_value2))

    # print(fc.weight.data.view(-1))



def predictAndCalculateRSq(allY, features, all_adj, trained_model,epoch):
    pred_y_total = []
    y_total = []

    # #a bunch of debug statements
    # print("allY value length (must be 331):")
    # print((allY.shape))

    # print("all_adj:")
    # print(all_adj)
    # print("each_adj value length (must be 331):")
    # print(len(all_adj))

    # print("features length (must be 331):")
    # print((features.shape))


    loss_fn = nn.MSELoss(size_average=True)

    adj_gold_pred={}
    previous_adj=""
    current_adj=""
    this_adj_gold_y=[]
    this_adj_pred_y=[]

    for index,feature in tqdm(enumerate(features), total=len(features), desc="predict:"):


            featureV= convert_to_variable(feature)
            y = allY[index]
            each_adj = all_adj[index]
            pred_y = trained_model(each_adj, featureV)
            y_total.append(y)
            pred_y_total.append(pred_y.data.cpu().numpy()[0])

            #for each data point which has the same adjective, store its goldY and predY values
            current_adj=each_adj

            #very first time initialize the previous_adj=current_adj
            if(index==0):
                previous_adj=current_adj
                # print("foujnd that index==0")

            if(current_adj==previous_adj):
                this_adj_gold_y.append(y)
                this_adj_pred_y.append(pred_y.data.cpu().numpy()[0])
                # print("foujnd that this adj and previous adj are same.")


            #if the adjectives are different, it means that we are switching to a new one. calculate rsquared. update previous_adj
            else:
                # print("foujnd that this adj and previous adj are NOT same.")
                # print(str(len(this_adj_gold_y)))
                # print(str(len(this_adj_pred_y)))
                previous_adj=current_adj
                rsquared_value_per_adj=r2_score(this_adj_gold_y, this_adj_pred_y, sample_weight=None, multioutput='uniform_average')

                if((epoch%5)==0):
                    print("adj:"+current_adj+" rsq value:"+str(rsquared_value_per_adj))

        #loss_dev = loss_fn(pred_y, true_variable_y)


    # print("allY value length (must be 331):")
    # print(len(y_total))
    # print("predicted allY value length (must be 331):")
    # print(len(pred_y_total))

    rsquared_value=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')
    return rsquared_value


'''splice the glove embeddings to get the embeddings for only the adjectives you need.'''
def cutGlove(adj_lexicon):
        print("going to load glove:")
        # load the glove embeddings for this adjective
        vocab, vec = torchwordemb.load_glove_text("/data/nlp/corpora/glove/6B/glove.6B.300d.txt")

        #for each unique adjective in teh training data, get its embedding and add it to another vector file

        adj_glove_emb={}
        # print("size of adj_lexicon")
        # print(len(adj_lexicon))
        for index, adj in enumerate(adj_lexicon):
            emb=vec[vocab[adj]]
            adj_glove_emb[adj]=emb


        # print("length of embeddings for all adj")
        #
        # print(len(adj_glove_emb))
        return adj_glove_emb


def tuneOnDev(trained_model,dev,cwd, uniq_turker,rsq_values,rsquared_value_training,loss_training,addTurkerOneHot,epoch):
    # test on dev data
    features, y, adj_lexicon, all_adj = get_features_dev(cwd, dev, False, uniq_turker,addTurkerOneHot)
    #print("done reading dev data:")

    # calculate rsquared
    rsquared_value = predictAndCalculateRSq(y, features, all_adj, trained_model,epoch)

    #print(str(loss_training)+"\t"+ str(rsquared_value))

    print("")
    print(str(rsquared_value_training)+"\t"+ str(rsquared_value))
    print("")
    rsq_values.write(str(rsquared_value)+"\n")

