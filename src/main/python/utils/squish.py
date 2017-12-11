import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab
import torchwordemb


from tqdm import tqdm


torch.manual_seed(1)

dense_size=10
no_of_epochs=1
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

        #the linear regression code which maps hidden layer to intercept value must come here


    #init was where you where just defining what embeddings meant. Here we actually use it
    def forward(self,adj):

        #get the corresponding  embeddings of the adjective
        emb_adj=self.embeddings(adj)




        #get the glove embeddings for this adjective
        vocab, vec = torchwordemb.load_glove_text("/data/nlp/corpora/glove/6B/glove.6B.300d.txt")
        print(vec.size())
        print("adj:")
        print(adj)
        emb=vec[vocab[adj]].numpy()
        embT =torch.from_numpy(emb)
        embV=Variable(embT,requires_grad=True)

        #give that to the squishing layer
        squished_layer=self.squish(embV,-1)

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


#the actual trainign code. Basically create an object of the class above
def run_adj_emb(list_Adj):
    #take the list of adjectives and give it all an index
    adj_index=convert_adj_index(list_Adj)

    print("got inside run_adj_emb. going to call model")
    model=AdjEmb()
    loss_function= nn.MSELoss(size_average=True)
    #rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)

    #run through each epoch, feed forward, calculate loss, back propagate
    for epoch in tqdm(range(no_of_epochs),total=no_of_epochs,desc="squishing:"):

        #for each word in the list of adjectives
        for each_adj in tqdm(list_Adj,total=len(list_Adj),desc="each_adj:"):

            print("got inside each_adj. going to call model.zero grad")

            model.zero_grad()

            print("value of each_adj is:"+str(each_adj))
            #convert adj into the right sequence
            adj_variable=getIndex(each_adj,adj_index)

            print("value of adj_variable is:"+str(adj_variable))

            squished_emb=model(adj_variable)

            #concatenate this squished embedding with turk one hot vector, and do linear regression
            #todo: call linear regression code here, or concatenate these vectors
            print(squished_emb)

            sys.exit(1)

            #calculate the loss
            loss=loss_function(squished_emb)

            loss.backward()
            #rms.step()




    print('Loss: after all epochs'+str((loss.data)))

    print("y value:")
    print(y)
    print("predicted y value")
    print(pred_y)
    rsquared_value=r2_score(y, pred_y, sample_weight=None, multioutput='uniform_average')


    print("rsquared_value:")
    print(str(rsquared_value))

    # #rsquared_value2= rsquared(y, pred_y)
    # print("rsquared_value2:")
    # print(str(rsquared_value2))

    # print(fc.weight.data.view(-1))
    # learned_weights = fc.weight.data
    # return(learned_weights.cpu().numpy())



