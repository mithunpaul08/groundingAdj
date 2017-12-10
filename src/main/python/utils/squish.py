import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import re
import pickle as pk
import torchwordemb
import os
torch.manual_seed(1)
import torchtext.vocab as vocab

noofEpochs=1
#create a dictionary (which will be filled later, to store the unique words and its indexes)
wordsAndIndices = {}

#useGlove=False;
#create a similar hash table for all the pos tags and give it an index
tagsAndIndices = {}


def prepare_sequence(seq, to_ix):
    idxs=[]
    # if the word doesnt exist return a a random number-this is a temporary fix
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            #print("found that this word wasnt seen during training:"+str(w))
            idxs.append(to_ix["the"])

    #print("idxs")
    #print(idxs)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)



def getIndex(w, to_ix):
    #idxs=[0, 1, 2, 3, 4]
    #print(w)
    idxs = [to_ix[w]]
    #print(idxs)
    #print("just printed indices")
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# def prepare_sequence():
#     idxs=[0, 1, 2, 3, 4]
#     #idxs = [to_ix[w] for w in seq]
#     tensor = torch.LongTensor(idxs)
#     return autograd.Variable(tensor)

#an array of sentences
sentence_collection=[]


def prepare_training_data(posTrain):

    for eachSent in tqdm(posTrain,total=len(posTrain),desc="tags_train:"):
        # print("eachSent:")
        # print(eachSent)
        for eachWord in eachSent[0]:
                #print(eachWord)
                if eachWord not in wordsAndIndices:
                    wordsAndIndices[eachWord] = len(wordsAndIndices)


    file_Name2 = "wordsAndIndices.pkl"
    # open the file for writing
    fileObject2 = open(file_Name2,'wb')
    pk.dump(wordsAndIndices, fileObject2)



def prepare_tags_data(posTrain):

    for eachSent in tqdm(posTrain,total=len(posTrain),desc="words_train :"):
        # print("eachSent:")
        # print(eachSent)
        for tag in eachSent[1]:
                if tag not in tagsAndIndices:
                    tagsAndIndices[tag] = len(tagsAndIndices)

    file_Name1 = "tagsAndIndices.pkl"
    # open the file for writing
    fileObject1 = open(file_Name1,'wb')
    pk.dump(tagsAndIndices, fileObject1)


#also create a hashtable from index to POS tag Eg: 1:NNP

indicesAndTags={}

def reverse_tags(localtagsAndIndices):
    for tag,index in tqdm(localtagsAndIndices.items(),total=len(localtagsAndIndices.items()),desc="tagRev :"):
        if index not in indicesAndTags:
                indicesAndTags[index] = (tag)
    return indicesAndTags

EMBEDDING_DIM = 300
HIDDEN_DIM = 10


cwd = os.getcwd()
path = cwd+"/data/glove/glove.6B.300d.txt"
#
class Squisher(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,useGlove):
        super(Squisher, self).__init__()
        self.hidden_dim = hidden_dim

        # #read teh glove data
        if(useGlove):
            print("going to load glove embeddings")
            glove = vocab.GloVe(name='6B', dim=300)
            # print('Loaded {} words'.format(len(glove.itos)))
            # print("glove.vectors.size(0)")
            # print(glove.vectors.size(0))
            # print("glove.vectors[0]")
            # print(glove.vectors[0])
            self.word_embeddings = nn.Embedding(glove.vectors.size(0), glove.vectors.size(1))
            self.word_embeddings.weight.data.copy_((glove.vectors))
            #self.word_embeddings = nn.Parameter(glove.vectors)
        else:
            print("no glove embeddings")
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # print("size of word embeddings now is:")
        # print((self.word_embeddings))




        #multiply weight with input vector
        affine=fc(batch_x)
        pred_y=affine.data.cpu().numpy()


        loss = loss_fn(affine, batch_y)


        # Backward pass
        loss.backward()


        #self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


def startSquisher(training_data, testData, gloveUseValue):
    #useGlove=gloveUseValue

    prepare_tags_data(training_data)
    prepare_training_data(training_data)


    model = Squisher(EMBEDDING_DIM, HIDDEN_DIM, len(wordsAndIndices), len(tagsAndIndices), gloveUseValue)
    #loss_function = nn.NLLLoss()
    loss_function = nn.MSELoss(size_average=True)


    #############add an optimizer optimizer.step()
        # adam.step()
        #optimizer = optim.SGD(fc.parameters(), lr=0.00000001)
        #adam = optim.Adam(fc.parameters())
    rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    optimizer.step()




    inputs = prepare_sequence(training_data[0][0], wordsAndIndices)
    tag_scores = model(inputs)


    for epoch in tqdm(range(noofEpochs),total=noofEpochs,desc="epochs_train:"):
        sentenceCounter=0
        for sentence,tags in tqdm(training_data ,total=len(training_data),desc="sent_train:"):



            model.zero_grad()
            model.hidden = model.init_hidden()

            sentence_in = prepare_sequence(sentence, wordsAndIndices)
            targets = prepare_sequence(tags, tagsAndIndices)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    ###############the model is trained by now
    file_Name5 = "lstm.pkl"
    # open the file for writing
    fileObject5 = open(file_Name5,'wb')
    pk.dump(model, fileObject5)

    ##########3testing phase

    correctlyPred=0;
    overallWordCounter=0
    sentenceCounter=0
    #for each sentences in the test data, feed it to the LSTM tagger
    for eachTuple in tqdm(testData,total=len(testData),desc="testData :"):


        sentenceCounter=sentenceCounter+1



        eachSent=eachTuple[0][0]

        listOfGoldTags=eachTuple[1][0]


        completePredTags=[]


        inputs = prepare_sequence(eachSent, wordsAndIndices)
        tag_scores = model(inputs)

        allTags=[]

        wordCounter=0

        #to map indices to tags
        localIndicesAndTags=reverse_tags(tagsAndIndices)


        for perWordScore in tag_scores:
            #print("perWordScore")
            #print(perWordScore)
            wordCounter=wordCounter+1
            overallWordCounter=overallWordCounter+1



            #print("word:"+str(eachSent[wordCounter-1]))




            highestScoreSoFar=99999

            highestIndex=0
            tagCounter=0;
            predicted_tag="NN"
            scoreCOunter=0
            #go through each of the scores of 45 pos tags and pick the highest
            for score in perWordScore:
                #print("score for this tag:")
                #print(score)

                scoreCOunter=scoreCOunter+1
                posvalue=score.data[0]*-1


                if(posvalue < highestScoreSoFar):
                        highestScoreSoFar=posvalue;
                        highestIndex=tagCounter

                tagCounter=tagCounter+1

            #after going through all the pos tags predicted print the higest score and highest inded

            #print("highestScoreSoFar")
            #print(highestScoreSoFar)
            #print("highestIndex:")
            #print(highestIndex)
            predTag=localIndicesAndTags[highestIndex]
            #print("predicted tag:"+predTag)
            completePredTags.append(predTag)

            #print("gold tag:"+str(listOfGoldTags[wordCounter]))

            if(predTag==listOfGoldTags[wordCounter]):
                correctlyPred=correctlyPred+1

            #print("predicted tags:"+str(completePredTags))
            # if(wordCounter>8):
            #     #print("word counter is more than 8")
            #     #print(wordCounter)
            #     sys.exit(1)



        #print("************done with all words in this sentence")
        # print("input gold tags:" + str(listOfGoldTags))
        # print("predicted tags:"+str(completePredTags))
        #
        #
        # predTagCounter=0
        # for eachPredTag in completePredTags:
        #     if(eachPredTag==listOfGoldTags[predTagCounter+1]):
        #         correctlyPred=correctlyPred+1
        #     predTagCounter=predTagCounter+1

    print("************done with all sentences")
    print("correctlyPred")
    print(correctlyPred)
    print("overallWordCounter")
    print(overallWordCounter)
    accuracy=((correctlyPred*100)/overallWordCounter)
    print("accuracy:"+str(accuracy)+" %")

