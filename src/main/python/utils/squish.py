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
from utils.grounding import get_features_labels_from_data
from utils.grounding import get_features_dev
torch.manual_seed(1)

#hidden_layers=[30,1]
# no_of_hidden_layers=3
dense1_size=1
#dense2_size=1
# dense3_size=1

noOfFoldsCV=4
noOfEpochs=1000
learning_rate=1e-5
patience_max=5;
#lr=1e-2

rsq_file="rsq_file.txt"
rsq_file_nfcv="rsq_file_nfcv.txt"
rsq_file_nfcv_avrg="rsq_file_nfcv_avrg.txt"
rsq_per_epoch_dev_four_chunks="rsq_per_epoch_dev_4_chunks.txt"

training_data="trainingData.csv"
#test_data="test.csv"
#test_data="test_no_random_seed.csv"
#test_data="test_rand_seed1.csv"
test_data="test_no_random_seed2.csv"

random_seed=1
useRandomSeed=True

class AdjEmb(nn.Module):
    #the constructor. Pass whatever you need to
    def __init__(self,turkCount,addTurkerOneHot):
        super(AdjEmb,self).__init__()


        # get teh glove vectors
        #print("going to load glove for per adj.")

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
        # print("just loaded glove for per adj. going to load glove for entire embeddings.")
        #
        # print(".self.vec.shape[0]")
        # print(self.vec.shape[0])
        # print(".self.vec.shape[1]")
        # print(self.vec.shape[1])
        self.embeddings = nn.Embedding(self.vec.shape[0], self.vec.shape[1])
        self.embeddings.weight.data.copy_(self.vec)

        #dont update embeddings
        self.embeddings.weight.requires_grad=False


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
            #print("found addTurkerOneHot=true")
        else:
            #use this when you dont have one hot for turkers
            self.fc = torch.nn.Linear(dense1_size+2, 1)



        #print("done loading all gloves")

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
    rms = optim.RMSprop(params_to_update, learning_rate, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
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
            rsq_values.flush()
            # print("loss_training:")
            # print(loss_training)





            #this is a hack. we need to put early stopping or something here
                #once you hit a good rsq value, break out of the epochs loop and save the model and run on test partition
            foundGoodModel=tuneOnDev(model,dev,cwd, uniq_turker,rsq_values,rsquared_value_training,loss_training,addTurkerOneHot,epoch)

            # if(foundGoodModel):
            #     break
            # Print weights
            learned_weights = model.fc.weight.data
            #print("\tlearned weights:" + str(learned_weights.cpu().numpy()))
            if(epoch==120):
                # the model is trained by now-store it to disk
                file_Name122 = "adj_data_80-10-10-120-epochs.pkl"
                # open the file for writing
                fileObject122 = open(file_Name122, 'wb')
                pk.dump(model, fileObject122)





    print("found a good model after tuning the model on dev. Going to  write it to disk")


    #the model is trained by now-store it to disk
    file_Name_200 = "adj_data_80-10-10-145epochs.pkl"
    # open the file for writing
    fileObject_200 = open(file_Name_200,'wb')
    pk.dump(model, fileObject_200)

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
def run_nfoldCV_on_turk_data_with_estopping(features, allY, uniq_adj, all_adj,addTurkerOneHot,useEarlyStopping,use4Chunks):
    # shuffle before splitting for early stopping
    np.random.seed(1)

    allIndex = np.arange(len(features))
    print("str(len(features)):")
    print(str(len(features)))



    #split it into folds. n=number of folds. almost even sized.
    n=noOfFoldsCV
    split_data=chunk(allIndex,n)
    #print(str(split_data))
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
        #note:eachChunkIndex starts at zero
        for eachChunkIndex in tqdm(chunkIndices,total=len(chunkIndices), desc="n-fold-CV:"):

            print("**************Starting next chunk, chunk number:"+str(eachChunkIndex)+" out of: "+str(len(chunkIndices)))

            model = AdjEmb(193, addTurkerOneHot)

            params_to_update = filter(lambda p: p.requires_grad == True, model.parameters())
            rms = optim.RMSprop(params_to_update, lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
            loss_fn = nn.MSELoss(size_average=True)


            '''experiment: out of 4 chunks, keep one for testing, one for dev, and the rest two as training'''

            if(use4Chunks):
                dev_chunk_index = (eachChunkIndex + 1) % 4

                # create a  list of all the indices of chunks except the test and dev chunk you are keeping out
                tr_data_chunk_indices = []
                for i in chunkIndices:
                    if (i != eachChunkIndex and i!=dev_chunk_index):
                        tr_data_chunk_indices.append(i)


            else:
                # create a  list of all the indices of chunks except the chunk you are keeping out
                tr_data_chunk_indices=[]
                for i in chunkIndices:
                    if i!=eachChunkIndex:
                        tr_data_chunk_indices.append(i)

            # print("tr_data_chunk_indices:" + str(tr_data_chunk_indices))
            # print("eachChunkIndex:" + str(eachChunkIndex))
            # print("dev_chunk_index:"+str(dev_chunk_index))



            training_data=[]

            #for each of these left over chunks, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the training data
            for eachChunk in tr_data_chunk_indices:
                for eachElement in split_data[eachChunk]:
                    training_data.append(eachElement)

            print("length of training_data:"+str(len(training_data)))
            test_data=[]

            #for the left out test chunk, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the test data
            for eachElement in split_data[eachChunkIndex]:
                    test_data.append(eachElement)

            print("length of test_data:" + str(len(test_data)))

            # for the left out dev chunk, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the test data
            dev_data = []
            for eachElement_dev in split_data[dev_chunk_index]:
                dev_data.append(eachElement_dev)

            print("length of dev_data:" + str(len(dev_data)))



            rsq_max_estop=0.000
            rsq_previous_estop=0.000
            patienceCounter=0;


            '''feed the LOOCV with custom data, and not random chunks. this is a temporary hack for sanity check. '''

            # # read the training data
            # training_data, y, adj_lexicon, all_adj, uniq_turker,uniq_adj_list = get_features_labels_from_data(cwd, training_data,
            #                                                                                                      addAdjOneHot, uniq_turker, addTurkerOneHot)
            #
            #
            # # read the test data
            # training_data, y, adj_lexicon, all_adj, uniq_turker,uniq_adj_list = get_features_labels_from_data(cwd, training_data,
            #                                                                                                      addAdjOneHot, uniq_turker, addTurkerOneHot)
            #
            #
            # print(training_data)
            # sys.exit(1)
            '''end of sanity check code'''

            np.random.shuffle(training_data)

            # print("size  of training_data1:" + str((len(training_data))))
            # print("size of  test_data:" + str((len(test_data))))

            '''adding early-stopping and patience'''
            if (useEarlyStopping):
                # split the training data further into training and dev
                len_training_estop = len(training_data)
                indices_tr_estop = np.arange(len_training_estop)
                limit_estop = math.ceil(len_training_estop * (50 / 100))
                trainingData_estop = indices_tr_estop[:limit_estop]
                dev_estop = indices_tr_estop[limit_estop:]
                training_data = trainingData_estop



                # debug statements
                # print("length of training estop:")
                # print(len(trainingData_estop))
                #
                # print("length of training estop:")
                # print(len(training_data))
                # # print("(trainingData_estop):")
                # # print((trainingData_estop))
                # # print("size of  len_training_estop:" + str((len_training_estop)))
                # print("size of  dev_estop:" + str(len(dev_estop)))




                # print("(training_data):")
                # print((training_data))

            #the patience counter starts from patience_max and decreases till it hits 0
            patienceCounter = patience_max

            #run n epochs on the left over training data
            for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):

                # # shuffle before each epoch
                np.random.shuffle(training_data)

                #print("size of  length of training_data2:" + str((len(training_data))))





                '''for each row in the training data, predict y value for itself, and then back
                propagate the loss'''
                for eachRow in tqdm(training_data, total=len(training_data), desc="trng_data_point:"):


                    #every time you feed forward, make sure the gradients are emptied out. From pytorch documentation
                    model.zero_grad()

                    feature=features[eachRow]

                    #print("feature:"+str(feature))

                    y = allY[eachRow]
                    each_adj = all_adj[eachRow]

                    #print("each_adj:"+str(each_adj))
                    #print("y:"+str(y))



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

                    #print("size of  dev_estop:" + str(len(dev_estop)))
                    pred_y_total_dev_data = []
                    y_total_dev_data = []

                    # for each element in the dev data, calculate its predicted value, and append it to predy_total
                    for dev_estop_index in dev_estop:
                        this_feature = features[dev_estop_index]
                        featureV_loo = convert_to_variable(this_feature)
                        y = allY[dev_estop_index]
                        each_adj = all_adj[dev_estop_index]
                        pred_y = model(each_adj, featureV_loo)
                        y_total_dev_data.append(y)
                        pred_y_total_dev_data.append(pred_y.data.cpu().numpy())

                    # calculate the rsquared value for entire dev_estop



                    # print("size of y_total_dev_data:"+str(len(y_total_dev_data)))
                    # print("size of pred_y_total_dev_data:" + str(len(pred_y_total_dev_data)))

                    rsquared_value_estop = r2_score(y_total_dev_data, pred_y_total_dev_data, sample_weight=None,
                                              multioutput='uniform_average')
                    # print("\n")
                    # print("rsquared_value_estop:" + str(rsquared_value_estop))
                    # print("\n")

                    #in the first epoch all the values are initialized to the current value
                    if(epoch==0):
                        rsq_max_estop = rsquared_value_estop
                        rsq_previous_estop = rsquared_value_estop

                    #2nd epoch onwards keep track of the maximum rsq value so far
                    else:

                        if(rsquared_value_estop > rsq_max_estop):
                            print("found that we have a new max value:"+str(rsquared_value_estop))
                            rsq_max_estop = rsquared_value_estop

                            # store the model to disk every time we hit a max.
                            # this is because at the end of hitting patience limit, we want the best model to test on the held out chunk
                            file_Name5 = "rsq_best_model_chunk_"+str(eachChunkIndex)+".pkl"
                            # open the file for writing
                            fileObject5 = open(file_Name5,'wb')
                            pk.dump(model, fileObject5)



                    #everytime the current rsquared value is less than the previous value, decrease patience count
                    if (rsquared_value_estop < rsq_previous_estop):
                        print("found that rsquared_value_estop is less than"
                              " rsq_previous_estop. going to increase patience:" )
                        patienceCounter=patienceCounter-1
                    else:
                        #increase the patience every time it gets a good value
                        patienceCounter = patienceCounter + 1
                        if(patienceCounter>patience_max):
                            patienceCounter=patience_max

                    print("epoch:"+str(epoch)+" rsq_max:"+str(rsq_max_estop)+" rsq_previous:"
                          +str(rsq_previous_estop) +" rsq_current:"+str(rsquared_value_estop)+
                          " patience:"+str(patienceCounter)+" loss:"+str(loss.data[0]))

                    rsq_previous_estop = rsquared_value_estop



                    if(patienceCounter < 1 ):
                        print("losing my patience. Have hit 0 . Exiting")
                        print("rsq_max_estop:"+str(rsq_max_estop))

                        #once patience runs out, load the model that was saved at the best max rsq value-and use that to test the held out chunk
                        trained_model_nfcv = pk.load(open("rsq_best_model_chunk_"+str(eachChunkIndex)+".pkl", "rb"))

                        #at the end of all epochs take the trained model that was trained on the 29 epochs
                        #and use the trained model to predict on the values in the left over chunk


                        pred_y_total_test_data = []
                        y_total_test_data = []


                        #for each element in the test data, calculate its predicted value, and append it to predy_total
                        #for test_data_index in dev_estop:
                        for test_data_index in test_data:
                            this_feature = features[test_data_index]
                            featureV_loo= convert_to_variable(this_feature)
                            y = allY[test_data_index]
                            each_adj = all_adj[test_data_index]
                            pred_y = trained_model_nfcv(each_adj, featureV_loo)
                            y_total_test_data.append(y)
                            pred_y_total_test_data.append(pred_y.data.cpu().numpy())



                        #calculate the rsquared value for this  held out
                        rsquared_value=r2_score(y_total_test_data, pred_y_total_test_data, sample_weight=None, multioutput='uniform_average')
                        print("\n")
                        print("rsquared_value_on_test_after_chunk_"+str(eachChunkIndex)+":"+str(rsquared_value))
                        print("\n")
                        nfcv.write(str(eachChunkIndex) + "\t" + str(rsquared_value) + "\n")
                        nfcv.flush()
                        rsq_total.append(rsquared_value)
                        break;


    #  After all chunks are done, calculate the average of each element in the list of predicted rsquared values.
    # There should be 10 such values,
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



'''  create feed forward NN model, but using 100 data points (around 33 folds) for cross validation'''
def run_nfoldCV_on_turk_data(features, allY, uniq_adj, all_adj,addTurkerOneHot,useEarlyStopping,use4Chunks):
    # shuffle before splitting for early stopping
    np.random.seed(1)

    allIndex = np.arange(len(features))
    print("str(len(features)):")
    print(str(len(features)))



    #split it into folds. n=number of folds. almost even sized.
    n=noOfFoldsCV
    split_data=chunk(allIndex,n)
    #print(str(split_data))
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
        #note:eachChunkIndex starts at zero
        for eachChunkIndex in tqdm(chunkIndices,total=len(chunkIndices), desc="n-fold-CV:"):

            print("**************Starting next chunk, chunk number:"+str(eachChunkIndex)+" out of: "+str(len(chunkIndices)))

            model = AdjEmb(193, addTurkerOneHot)

            params_to_update = filter(lambda p: p.requires_grad == True, model.parameters())
            rms = optim.RMSprop(params_to_update, lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
            loss_fn = nn.MSELoss(size_average=True)


            '''experiment: out of 4 chunks, keep one for testing, one for dev, and the rest two as training'''

            if(use4Chunks):
                dev_chunk_index = (eachChunkIndex + 1) % 4

                # create a  list of all the indices of chunks except the test and dev chunk you are keeping out
                tr_data_chunk_indices = []
                for i in chunkIndices:
                    if (i != eachChunkIndex and i!=dev_chunk_index):
                        tr_data_chunk_indices.append(i)


            else:
                # create a  list of all the indices of chunks except the chunk you are keeping out
                tr_data_chunk_indices=[]
                for i in chunkIndices:
                    if i!=eachChunkIndex:
                        tr_data_chunk_indices.append(i)

            # print("tr_data_chunk_indices:" + str(tr_data_chunk_indices))
            # print("eachChunkIndex:" + str(eachChunkIndex))
            # print("dev_chunk_index:"+str(dev_chunk_index))



            training_data=[]

            #for each of these left over chunks, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the training data
            for eachChunk in tr_data_chunk_indices:
                for eachElement in split_data[eachChunk]:
                    training_data.append(eachElement)

            print("length of training_data:"+str(len(training_data)))
            test_data=[]

            #for the left out test chunk, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the test data
            for eachElement in split_data[eachChunkIndex]:
                    test_data.append(eachElement)

            print("length of test_data:" + str(len(test_data)))

            # for the left out dev chunk, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the test data
            dev_data = []
            for eachElement_dev in split_data[dev_chunk_index]:
                dev_data.append(eachElement_dev)

            print("length of dev_data:" + str(len(dev_data)))



            rsq_max_estop=0.000
            rsq_previous_estop=0.000
            patienceCounter=0;


            '''feed the LOOCV with custom data, and not random chunks. this is a temporary hack for sanity check. '''

            # # read the training data
            # training_data, y, adj_lexicon, all_adj, uniq_turker,uniq_adj_list = get_features_labels_from_data(cwd, training_data,
            #                                                                                                      addAdjOneHot, uniq_turker, addTurkerOneHot)
            #
            #
            # # read the test data
            # training_data, y, adj_lexicon, all_adj, uniq_turker,uniq_adj_list = get_features_labels_from_data(cwd, training_data,
            #                                                                                                      addAdjOneHot, uniq_turker, addTurkerOneHot)
            #
            #
            # print(training_data)
            # sys.exit(1)
            '''end of sanity check code'''

            np.random.shuffle(training_data)

            # print("size  of training_data1:" + str((len(training_data))))
            # print("size of  test_data:" + str((len(test_data))))

            '''adding early-stopping and patience'''



                # debug statements
                # print("length of training estop:")
                # print(len(trainingData_estop))
                #
                # print("length of training estop:")
                # print(len(training_data))
                # # print("(trainingData_estop):")
                # # print((trainingData_estop))
                # # print("size of  len_training_estop:" + str((len_training_estop)))
                # print("size of  dev_estop:" + str(len(dev_estop)))




                # print("(training_data):")
                # print((training_data))

            #the patience counter starts from patience_max and decreases till it hits 0
            patienceCounter = patience_max

            #run n epochs on the left over training data
            for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):

                # # shuffle before each epoch
                np.random.shuffle(training_data)

                #print("size of  length of training_data2:" + str((len(training_data))))





                '''for each row in the training data, predict y value for itself, and then back
                propagate the loss'''
                for eachRow in tqdm(training_data, total=len(training_data), desc="trng_data_point:"):


                    #every time you feed forward, make sure the gradients are emptied out. From pytorch documentation
                    model.zero_grad()

                    feature=features[eachRow]

                    #print("feature:"+str(feature))

                    y = allY[eachRow]
                    each_adj = all_adj[eachRow]

                    #print("each_adj:"+str(each_adj))
                    #print("y:"+str(y))



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

                    #print("size of  dev_estop:" + str(len(dev_estop)))
                    pred_y_total_dev_data = []
                    y_total_dev_data = []

                    # for each element in the dev data, calculate its predicted value, and append it to predy_total
                    for dev_estop_index in dev_estop:
                        this_feature = features[dev_estop_index]
                        featureV_loo = convert_to_variable(this_feature)
                        y = allY[dev_estop_index]
                        each_adj = all_adj[dev_estop_index]
                        pred_y = model(each_adj, featureV_loo)
                        y_total_dev_data.append(y)
                        pred_y_total_dev_data.append(pred_y.data.cpu().numpy())

                    # calculate the rsquared value for entire dev_estop



                    # print("size of y_total_dev_data:"+str(len(y_total_dev_data)))
                    # print("size of pred_y_total_dev_data:" + str(len(pred_y_total_dev_data)))

                    rsquared_value_estop = r2_score(y_total_dev_data, pred_y_total_dev_data, sample_weight=None,
                                              multioutput='uniform_average')
                    # print("\n")
                    # print("rsquared_value_estop:" + str(rsquared_value_estop))
                    # print("\n")

                    #in the first epoch all the values are initialized to the current value
                    if(epoch==0):
                        rsq_max_estop = rsquared_value_estop
                        rsq_previous_estop = rsquared_value_estop

                    #2nd epoch onwards keep track of the maximum rsq value so far
                    else:

                        if(rsquared_value_estop > rsq_max_estop):
                            print("found that we have a new max value:"+str(rsquared_value_estop))
                            rsq_max_estop = rsquared_value_estop

                            # store the model to disk every time we hit a max.
                            # this is because at the end of hitting patience limit, we want the best model to test on the held out chunk
                            file_Name5 = "rsq_best_model_chunk_"+str(eachChunkIndex)+".pkl"
                            # open the file for writing
                            fileObject5 = open(file_Name5,'wb')
                            pk.dump(model, fileObject5)



                    #everytime the current rsquared value is less than the previous value, decrease patience count
                    if (rsquared_value_estop < rsq_previous_estop):
                        print("found that rsquared_value_estop is less than"
                              " rsq_previous_estop. going to increase patience:" )
                        patienceCounter=patienceCounter-1
                    else:
                        #increase the patience every time it gets a good value
                        patienceCounter = patienceCounter + 1
                        if(patienceCounter>patience_max):
                            patienceCounter=patience_max

                    print("epoch:"+str(epoch)+" rsq_max:"+str(rsq_max_estop)+" rsq_previous:"
                          +str(rsq_previous_estop) +" rsq_current:"+str(rsquared_value_estop)+
                          " patience:"+str(patienceCounter)+" loss:"+str(loss.data[0]))

                    rsq_previous_estop = rsquared_value_estop



                    if(patienceCounter < 1 ):
                        print("losing my patience. Have hit 0 . Exiting")
                        print("rsq_max_estop:"+str(rsq_max_estop))

                        #once patience runs out, load the model that was saved at the best max rsq value-and use that to test the held out chunk
                        trained_model_nfcv = pk.load(open("rsq_best_model_chunk_"+str(eachChunkIndex)+".pkl", "rb"))

                        #at the end of all epochs take the trained model that was trained on the 29 epochs
                        #and use the trained model to predict on the values in the left over chunk


                        pred_y_total_test_data = []
                        y_total_test_data = []


                        #for each element in the test data, calculate its predicted value, and append it to predy_total
                        #for test_data_index in dev_estop:
                        for test_data_index in test_data:
                            this_feature = features[test_data_index]
                            featureV_loo= convert_to_variable(this_feature)
                            y = allY[test_data_index]
                            each_adj = all_adj[test_data_index]
                            pred_y = trained_model_nfcv(each_adj, featureV_loo)
                            y_total_test_data.append(y)
                            pred_y_total_test_data.append(pred_y.data.cpu().numpy())



                        #calculate the rsquared value for this  held out
                        rsquared_value=r2_score(y_total_test_data, pred_y_total_test_data, sample_weight=None, multioutput='uniform_average')
                        print("\n")
                        print("rsquared_value_on_test_after_chunk_"+str(eachChunkIndex)+":"+str(rsquared_value))
                        print("\n")
                        nfcv.write(str(eachChunkIndex) + "\t" + str(rsquared_value) + "\n")
                        nfcv.flush()
                        rsq_total.append(rsquared_value)
                        break;


    #  After all chunks are done, calculate the average of each element in the list of predicted rsquared values.
    # There should be 10 such values,
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



'''experiment: out of 4 chunks, keep one for testing, one for dev, and the rest two as training'''
def run_nfoldCV_on_turk_data_4chunks(features, allY, uniq_adj, all_adj,addTurkerOneHot,useEarlyStopping,use4Chunks):
    # shuffle before splitting
    if (useRandomSeed):
        np.random.seed(random_seed)



    allIndex = np.arange(len(features))
    print("str(len(features)):")
    print(str(len(features)))

    np.random.shuffle(allIndex)





    #split it into folds. n=number of folds. almost even sized.
    n=noOfFoldsCV
    split_data=chunk(allIndex,n)
    #print(str(split_data))
    chunkIndices = np.arange(len(split_data))


    rsq_total=[]
    cwd=os.getcwd()

    # write rsq per chunk to disk
    with open(cwd + "/outputs/" + rsq_file_nfcv, "w+")as nfcv:
        #empty out the existing file before loop does append
        nfcv.write("Chunk \t RSQ\n")
        nfcv.close()

        # tp write rsq per epoch  to disk
        # first empty out the existing file before loop does append
    with open(cwd + "/outputs/" + rsq_per_epoch_dev_four_chunks, "w+")as nfcv_four:
        nfcv_four.write("")
        nfcv_four.close()


    with open(cwd + "/outputs/" + rsq_file_nfcv, "a")as nfcv:

        # for each chunk in the training data, keep that one out, and train on the rest
        # append the rest of the values
        #note:test_fold_index starts at zero
        for test_fold_index in tqdm(chunkIndices,total=len(chunkIndices), desc="n-fold-CV:"):

            #left over from an earlier hack. too lazy to tab like 1000 lines
            if(test_fold_index!=0):



                print("**************Starting next fold, fold number:"+str(test_fold_index)+" out of: "+str(len(chunkIndices))+"\n")

                model_4chunk = AdjEmb(193, addTurkerOneHot)

                params_to_update = filter(lambda p: p.requires_grad == True, model_4chunk.parameters())
                rms = optim.RMSprop(params_to_update, lr=learning_rate, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
                loss_fn = nn.MSELoss(size_average=True)

                dev_fold_index = (test_fold_index + 1) % 4

                # create a  list of all the indices of chunks except the test and dev chunk you are keeping out
                tr_fold_indices = []
                for i in chunkIndices:
                    if (i != test_fold_index and i != dev_fold_index):
                        tr_fold_indices.append(i)



                # print("tr_fold_indices:" + str(tr_fold_indices))
                # print("test_fold_index:" + str(test_fold_index))
                # print("dev_fold_index:"+str(dev_fold_index))



                training_data=[]

                #for each of these left over chunks, pull out its data points, and concatenate all into one single huge list of
                # data points-this is the training data
                for eachChunk in tr_fold_indices:
                    for eachElement in split_data[eachChunk]:
                        training_data.append(eachElement)

                #print("length of training_data:"+str(len(training_data)))
                test_data=[]

                #for the left out test chunk, pull out its data points, and concatenate all into one single huge list of
                # data points-this is the test data
                for eachElement in split_data[test_fold_index]:
                        test_data.append(eachElement)

                #print("length of test_data:" + str(len(test_data)))

                # for the left out dev chunk, pull out its data points, and concatenate all into one single huge list of
                # data points-this is the test data
                dev_data = []
                for eachElement_dev in split_data[dev_fold_index]:
                    dev_data.append(eachElement_dev)

                uniqAdj_dev={}
                uniqAdj_test={}
                uniqAdj_training={}
                for eachDev in dev_data:
                    each_adj_tr = all_adj[eachDev]
                    uniqAdj_dev[each_adj_tr] = uniqAdj_dev.get(each_adj_tr, 0) + 1

                for eachDev in test_data:
                    each_adj_tr = all_adj[eachDev]
                    uniqAdj_test[each_adj_tr] = uniqAdj_test.get(each_adj_tr, 0) + 1

                for eachDev in training_data:
                    each_adj_tr = all_adj[eachDev]
                    uniqAdj_training[each_adj_tr] = uniqAdj_training.get(each_adj_tr, 0) + 1


                for (k,v) in uniqAdj_dev.items():
                    if k not in uniqAdj_training:
                       print("WARNING: " + k+" this adj from dev was not there in training")
                    # else:
                    #     print("\t"+k+" this adj from dev was present there in training")

                for (k,v) in uniqAdj_test.items():
                    if k not in uniqAdj_training:
                       print("WARNING: " + k+" this adj from test was not there in training")
                    # else:
                    #     print("\t"+k+" this adj from test was present there in training")

                # print("\nADJECTIVES:")
                # print("TRAINING:")
                # print(uniqAdj_training.items())
                #
                #
                # print("\nDEV:")
                # print(uniqAdj_dev.items())
                # print("\nTEST:")
                # print(uniqAdj_test.items())




                #print("length of dev_data:" + str(len(dev_data)))







                # print("size  of training_data1:" + str((len(training_data))))
                # print("size of  test_data:" + str((len(test_data))))





                    # print("(training_data):")
                    # print((training_data))

                #the patience counter starts from patience_max and decreases till it hits 0
                patienceCounter = patience_max



                #run n epochs on the left over training data
                with open(cwd + "/outputs/" + rsq_per_epoch_dev_four_chunks, "a")as nfcv_four:
                    nfcv_four.write("test_fold_index:" + str(test_fold_index)+"\n")
                    nfcv_four.write("dev_fold_index:"+str(dev_fold_index)+"\n")
                    nfcv_four.write("tr_fold_indices:" + str(tr_fold_indices) + "\n")
                    nfcv_four.write("Epoch \t RSQ_tr  \t RSQ_dev\n")

                    '''found the best epochs per fold. after tuning on dev'''
                    if(test_fold_index==0):
                        noOfEpochs=1947
                    else:
                        if(test_fold_index==1):
                            noOfEpochs=899
                        else:
                            if(test_fold_index==2):
                                noOfEpochs=990
                            else:
                                if(test_fold_index==3):
                                    noOfEpochs=983

                    for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):

                        y_total_tr_data=[]
                        pred_y_total_tr_data=[]

                        # # shuffle before each epoch
                        np.random.shuffle(training_data)

                        #print(training_data)


                        #print("size of  length of training_data2:" + str((len(training_data))))





                        '''for each row in the training data, predict y_test value for itself, and then back
                        propagate the loss'''
                        for each_data_item_index in tqdm(training_data, total=len(training_data), desc="trng_data_point:"):


                            #every time you feed forward, make sure the gradients are emptied out. From pytorch documentation
                            model_4chunk.zero_grad()

                            feature=features[each_data_item_index]
                            y_test = allY[each_data_item_index]
                            each_adj_tr = all_adj[each_data_item_index]

                            # print("feature:"+str(feature))
                            # print("each_adj_tr:"+str(each_adj_tr)+"\n")
                            # print("y_test:"+str(y_test))





                            featureV= convert_to_variable(feature)
                            pred_y_training = model_4chunk(each_adj_tr, featureV)

                            y_total_tr_data.append(y_test)
                            pred_y_total_tr_data.append(pred_y_training.data.cpu().numpy())


                            batch_y = convert_scalar_to_variable(y_test)

                            loss = loss_fn(pred_y_training, batch_y)


                            # Backward pass
                            loss.backward()

                            rms.step()




                        #after every epoch, i.e after training on n data points,-calculate rsq for trainign also
                        #rsquared_value_tr = r2_score(y_total_tr_data, pred_y_total_tr_data, sample_weight=None,
                                                  #multioutput='uniform_average')

                        # #after every epoch, i.e after training on n data points,
                        #  run on dev data and calculate rsq
                        # print("size of  dev_estop:" + str(len(dev_estop)))
                        pred_y_total_dev_data = []
                        y_total_dev_data = []

                        # print(dev_data)
                        # print(str(len(dev_data)))

                        # for each element in the dev data, calculate its predicted value, and append it to predy_total
                        # for dev_index in dev_data:
                        #     this_feature = features[dev_index]
                        #     featureV_dev = convert_to_variable(this_feature)
                        #     y_dev = allY[dev_index]
                        #     each_adj_dev = all_adj[dev_index]
                        #
                        #
                        #
                        #     pred_y_dev = model_4chunk(each_adj_dev, featureV_dev)
                        #     y_total_dev_data.append(y_dev)
                        #     pred_y_total_dev_data.append(pred_y_dev.data.cpu().numpy())

                            # print("feature:" + str(feature))
                            # print("each_adj_tr:" + str(each_adj_tr) + "\n")
                            # print("y_test:" + str(y_test))
                            # print(pred_y_training)




                        # print(y_total_dev_data)
                        # print(pred_y_total_dev_data)
                        # print("size of y_total_dev_data:"+str(len(y_total_dev_data)))
                        # print("size of pred_y_total_dev_data:" + str(len(pred_y_total_dev_data)))

                        # rsquared_value_dev = r2_score(y_total_dev_data, pred_y_total_dev_data, sample_weight=None,
                        #                           multioutput='uniform_average')
                        #
                        #
                        # # print("\n")
                        # # print("rsquared_value_Dev" + str(test_fold_index) + ":" + str(rsquared_value_dev))
                        # # print("\n")
                        #
                        # nfcv_four.write(str(epoch) + "\t" + str(rsquared_value_tr) +"\t" + str(rsquared_value_dev ) + "\n")
                        # nfcv_four.flush()







            print("done with all epochs")



            #Testing phase
            # after all epochs in the given chunk, (i.e test once per fold)
            # for each element in the test data, calculate its predicted value, and append it to predy_total

            y_total_test_data=[]
            pred_y_total_test_data=[]

            for test_data_index in test_data:
                this_feature = features[test_data_index]
                featureV_dev= convert_to_variable(this_feature)
                y_test = allY[test_data_index]
                each_adj_test = all_adj[test_data_index]
                pred_y_test = model_4chunk(each_adj_test, featureV_dev)
                y_total_test_data.append(y_test)
                pred_y_total_test_data.append(pred_y_test.data.cpu().numpy())



            #calculate the rsquared value for this  held out
            rsquared_value_test=r2_score(y_total_test_data, pred_y_total_test_data, sample_weight=None, multioutput='uniform_average')
            print("\n")
            print("rsquared_value_on_test_after_chunk_"+str(test_fold_index)+":"+str(rsquared_value_test))
            print("\n")
            nfcv.write(str(test_fold_index) + "\t" + str(rsquared_value_test) + "\n")
            nfcv.flush()
            rsq_total.append(rsquared_value_test)



    #  After all chunks are done, calculate the average of each element in the list of predicted rsquared values.
    # There should be 10 such values,
    # each corresponding to one chunk being held out


    print("done with all chunks")

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

                # if((epoch%5)==0):
                #     print("adj:"+current_adj+" rsq value:"+str(rsquared_value_per_adj))

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
    rsquared_dev_value = predictAndCalculateRSq(y, features, all_adj, trained_model,epoch)

    #print(str(loss_training)+"\t"+ str(rsquared_value))

    print("")
    print("rsquared_value_training:\n")
    print(str(rsquared_value_training))
    print("rsquared_value_dev:\n")
    print(str(rsquared_dev_value))
    print("")
    rsq_values.write(str(rsquared_dev_value)+"\n")
    rsq_values.flush()

    #this is a hack. we need to put early stopping or something here
    #once you hit a good rsq value, break and save the model and run on test partition
    # if(rsquared_dev_value>0.43):
    #     return True;


def runOnTestPartition(trained_model,dev,cwd, uniq_turker,rsq_values,addTurkerOneHot,epoch):
    # read the test
    features, y, adj_lexicon, all_adj = get_features_dev(cwd, dev, False, uniq_turker,addTurkerOneHot)
    print("done reading test data:")

    # calculate rsquared
    rsquared_test_value = predictAndCalculateRSq(y, features, all_adj, trained_model,epoch)

    #print(str(loss_training)+"\t"+ str(rsquared_value))


    print("rsquared_value_on_test:\n")
    print(str(rsquared_test_value))
    print("")
    rsq_values.write(str(rsquared_test_value)+"\n")
    rsq_values.flush()

