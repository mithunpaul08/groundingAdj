import pickle as pk
from sklearn.metrics import r2_score
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab
import torchwordemb
from utils.linearReg import convert_variable
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

torch.manual_seed(1)

dense1_size=20
dense2_size=10
dense3_size=1
noOfEpochs=100
class AdjEmb(nn.Module):
    #the constructor. Pass whatever you need to
    def __init__(self,turkCount):
        super(AdjEmb,self).__init__()

        # get the glove embeddings for this adjective
        self.vocab, self.vec = torchwordemb.load_glove_text("/data/nlp/corpora/glove/6B/glove.6B.300d.txt")
        self.noOfTurkers=turkCount

        # get teh glove vectors
        print("just loaded glove for per adj. going to load glove for entire embeddings.")

        print(".self.vec.shape[0]")
        print(self.vec.shape[0])
        print(".self.vec.shape[1]")
        print(self.vec.shape[1])
        self.embeddings = nn.Embedding(self.vec.shape[0], self.vec.shape[1])
        self.embeddings.weight.data.copy_(self.vec)
        self.embeddings.weight.requires_grad=False
        # the layer where you squish the 300 embeddings to a dense layer of 10
        # i.e it takes embeddings as input and returns a dense layer of size 10
        # note: this is also known as the weight vector to be used in an affine
        self.linear1 = nn.Linear(self.vec.size(1), dense1_size)
        #self.tanned=nn.Tanh(self.squish)
        self.linear2 = torch.nn.Linear(dense1_size, dense2_size)
        self.linear3 = torch.nn.Linear(dense2_size, dense3_size)
        self.fc = torch.nn.Linear(dense3_size+turkCount+2, 1)



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


        out=F.tanh(self.linear1(embV))
        out=F.tanh(self.linear2(out))
        out=F.tanh(self.linear3(out))





        feature_squished = torch.cat((feats, out))
        return self.fc(feature_squished)



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
def run_adj_emb(features, allY, list_Adj, all_adj):
    #take the list of adjectives and give it all an index
    adj_index=convert_adj_index(list_Adj)

    print("got inside run_adj_emb. going to Load Glove:")

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



    print("done with all training data")


    #the model is trained by now-store it to disk
    file_Name5 = "squish.pkl"
    # open the file for writing
    fileObject5 = open(file_Name5,'wb')
    pk.dump(model, fileObject5)

    learned_weights = fc.weight.data
    #return(learned_weights.cpu().numpy())


   #save the weights to disk
    file_Name1 = "learned_weights.pkl"
    # open the file for writing
    fileObject1 = open(file_Name1,'wb')
    pk.dump(learned_weights.cpu().numpy(), fileObject1)

    return model



    print("loss")

    #print(loss)
    #
    #
    # #todo: return the entire new 98x10 hashtable to regression code
    # print(adj_10_emb)
    # sys.exit(1)
    #
    # print('Loss: after all epochs'+str((loss.data)))
    #
    #print("allY value:")
    #print(len(y_total))
    #print("predicted allY value")
    #print(len(pred_y_total))
    rsquared_value=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')


    print("rsquared_value:")
    print(str(rsquared_value))
    #learned_weights = model.affine.weight.data
    #return(learned_weights.cpu().numpy())

    # #rsquared_value2= rsquared(allY, pred_y)
    # print("rsquared_value2:")
    # print(str(rsquared_value2))

    # print(fc.weight.data.view(-1))



def run_adj_emb_loocv(features, allY, list_Adj, all_adj):
    ''' same create feed forward NN model, but using loocv for cross validation'''

    #take the list of adjectives and give it all an index
    adj_index=convert_adj_index(list_Adj)

    print("got inside run_adj_emb. going to Load Glove:")

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




    pred_y_total = []
    y_total = []
    adj_10_emb = {}

    # do loocv len(trainingData) times
    for eachElement in tqdm(allIndex,total=len(allIndex), desc="eachTrngData:"):

        #for each element in the training data, keep that one out, and train on the rest
        #i.e create a list of all the indices except the one you are keeping out
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



def calculateRSq(features):
    for eachRow in tqdm(allIndex, total=len(features), desc="predict:"):
            feature=features[eachRow]
            y = allY[eachRow]
            each_adj = all_adj[eachRow]
            featureV= convert_to_variable(feature)
            pred_y = model(each_adj, featureV)
            y_total.append(y)
            pred_y_total.append(pred_y.data.cpu().numpy()[0])


    rsquared_value=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')
    return rsquared_value
