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

dense_size=10
noOfEpochs=10000
class AdjEmb(nn.Module):
    #the constructor. Pass whatever you need to
    def __init__(self):
        super(AdjEmb,self).__init__()

        #get teh glove vectors
        glove = vocab.GloVe(name='6B', dim=300)
        self.embeddings=nn.Embedding(glove.vectors.size(0),glove.vectors.size(1))

        #the layer where you squish the 300 embeddings to a dense layer of 10
        #i.e it takes embeddings as input and returns a dense layer of size 10
        #note: this is also known as the weight vector to be used in an affine
        self.squish=nn.Linear(glove.vectors.size(1),dense_size)
        #get the glove embeddings for this adjective
        self.vocab, self.vec = torchwordemb.load_glove_text("/data/nlp/corpora/glove/6B/glove.6B.300d.txt")

        #the linear regression code which maps hidden layer to intercept value must come here


    #init was where you where just defining what embeddings meant. Here we actually use it
    def forward(self,adj):

        #get the corresponding  embeddings of the adjective
        #emb_adj=self.embeddings(adj)





        #print(self.vec.size())
        #print("adj:")
        #print(adj)
        emb=self.vec[self.vocab[adj]].numpy()
        embT =torch.from_numpy(emb)
        embV=Variable(embT,requires_grad=False)

        #give that to the squishing layer
        squished_layer=self.squish(embV)

        return squished_layer



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

    return Variable(x2,requires_grad=False)

def convert_to_variable(features):

    x2 =torch.from_numpy(features)

    return Variable(x2,requires_grad=False)

#the actual trainign code. Basically create an object of the class above
def run_adj_emb(features, allY, list_Adj, all_adj):
    #take the list of adjectives and give it all an index
    adj_index=convert_adj_index(list_Adj)

    print("got inside run_adj_emb. going to Load Glove:")
    model=AdjEmb()

    #rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)

    #run through each epoch, feed forward, calculate loss, back propagate

    #no point having epoch if you are not back propagating
    #for epoch in tqdm(range(no_of_epochs),total=no_of_epochs,desc="squishing:"):

    #things needed for the linear regression phase
    featureShape=features.shape
    fc = torch.nn.Linear(205,1)
    rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    loss_fn = nn.MSELoss(size_average=True)


    for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):
        #for each word in the list of adjectives
        pred_y_total=[]
        y_total=[]
        adj_10_emb={}
        for feature, y, each_adj in tqdm((zip(features, allY, all_adj)), total=len(features), desc="each_adj:"):

            #print("got inside each_adj. going to call model.zero grad")

            #model.zero_grad()

            #print("value of each_adj is:"+str(each_adj))
            #convert adj into the right sequence
            #adj_variable=getIndex(each_adj,adj_index)

            #print("value of adj_variable is:"+str(adj_variable))

            squished_emb=model(each_adj)
            #print("squished_emb")
            #print(squished_emb)
            squished_np=squished_emb.data.numpy()

            #concatenate this squished embedding with turk one hot vector, and do linear regression

            featureV= convert_to_variable(feature)

            #print("feature")
            #print(featureV)

            #combined=np.concatenate(feature,squished_np)
            feature_squished=torch.cat((featureV,squished_emb.data))

            #print("feature_squished:")
            #print(feature_squished)

            batch_x=feature_squished






            adj_10_emb[each_adj]=squished_emb


            #the complete linear regression code- only thing is features here will include the squished_emb
            # Reset gradients
            fc.zero_grad()

            batch_y = convert_scalar_to_variable(y)
            y_total.append(y)

            loss_fn = nn.MSELoss(size_average=True)
            rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
            #print("batch_x")
            #print(batch_x)

            #multiply weight with input vector
            affine=fc(batch_x)

            #this is the actual prediction of the intercept
            pred_y=affine.data.cpu().numpy()
            pred_y_total.append(pred_y)




            loss = loss_fn(affine, batch_y)




            # Backward pass
            loss.backward()



            # optimizer.step()
            # adam.step()
            rms.step()



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



    print("loss")

    print(loss)
    #
    #
    # #todo: return the entire new 98x10 hashtable to regression code
    # print(adj_10_emb)
    # sys.exit(1)
    #
    # print('Loss: after all epochs'+str((loss.data)))
    #
    print("allY value:")
    print(len(y_total))
    print("predicted allY value")
    print(len(pred_y_total))
    rsquared_value=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')


    print("rsquared_value:")
    print(str(rsquared_value))
    return(learned_weights.cpu().numpy())

    # #rsquared_value2= rsquared(allY, pred_y)
    # print("rsquared_value2:")
    # print(str(rsquared_value2))

    # print(fc.weight.data.view(-1))




