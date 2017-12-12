import nltk as nlt
import os
import string
import sys
from collections import Counter
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from basefunctions import writeHighFreqTermsToFile
from sample import load_model, readFromDisk, pickle_file_name_train, base_path_output, base_path_test, \
    sentence_file_name_train_neg, sentence_file_name_train_pos
from wordembedtensor import kerasTokenizerUnit

MAX_SENTENCE_LENGTH=1000
topbestwords=1000



def predictresp(model, finalSequenceUnitTokenizer,norm_text):
    sequences = finalSequenceUnitTokenizer.texts_to_sequences(norm_text)
    finalSequenceUnit = pad_sequences(sequences, maxlen=MAX_SENTENCE_LENGTH, padding='pre')
    model_resp_prob = model.predict(finalSequenceUnit, verbose=0)
    model_resp = model.predict_classes(finalSequenceUnit, verbose=0)
    i = zip(model_resp, model_resp_prob)
    return i

def readFilesWithSentences(path, howManyFiles,global_nature):
    fileNames = os.listdir(os.path.abspath(path))
    fileContents = []
    fileRatings=[]
    a = 1
    # print(fileNames[0:howManyFiles])
    GroupLabel=[]
    for current in fileNames[0:howManyFiles]:
        with open(path + "//" + current, 'r', encoding="utf8") as openFile:
            sys.stdout.write('\r')
            sys.stdout.write("%d[%-100s]%d  %d%%" % (a,'=' * int(round((a/howManyFiles)*100,0)),howManyFiles, int(round((a/howManyFiles)*100,0))))
            sys.stdout.flush()
        #     # print("Currently Reading File : " + currentFile + " .Poll Progress:" + "(" + str(a) + " of " + str(
            readContent=openFile.readline()
            fileContents.append(readContent)

            stopWords = stopwords.words('english').remove('not')
            stop = sorted(
                stopWords + list(string.punctuation) + ["i\\", "'m", "'s", "it\\", '...', "''", '``', 'br',
                                                        's', '--'])
            tokens = nlt.tokenize.sent_tokenize(str(fileContents).lower())
            tokens = [w for w in tokens if w not in stop]

            abc=' '.join(tokens)
            # global GroupLabel
            rating=int(current[current.find("_")+1:current.find(".txt")])
            GroupLabel.append(abc)
            fileRatings.append(rating)
            a += 1
    print(global_nature+ " Files Read: %d" % howManyFiles,end="\n")
    return str(fileContents),GroupLabel,fileNames,fileRatings

def processInputTest():
    outputResult = open(base_path_output + "/" + "outputResultTrain.txt", 'w', encoding="utf8")
    positiveCorpus, positiveVectDict, wordFreqPos, GroupLabelPos, fileNamesPos, fileRatingsPos = writeHighFreqTermsToFile(
        base_path_test + "pos/", outputResult, "Positive")
    negativeCorpus, negativeVectDict, wordFreqNeg, GroupLabelNeg, fileNamesNeg, fileRatingsNeg = writeHighFreqTermsToFile(
        base_path_test + "neg/", outputResult, "Negative")
    outputResult.close()

    allGroupKeys = GroupLabelPos + GroupLabelNeg
    allGroupValues = fileRatingsPos + fileRatingsNeg
    allFileNames = fileNamesPos + fileNamesNeg
    allFileRatings = fileRatingsPos + fileRatingsNeg
    # print(saveToDisk(allGroupKeys, allGroupValues, allFileNames, allFileRatings,fileName=pickle_file_name_test))
    return allGroupKeys, allGroupValues, allFileNames, allFileRatings

model=load_model()

print("Reading from Pickle Object Saved.",end="\n")
allGroupKeysTrain, allGroupValuesTrain, allFileNamesTrain, allFileRatingsTrain=readFromDisk(pickle_file_name_train)
finalSequenceUnitTokenizer = kerasTokenizerUnit(allGroupKeysTrain, MAX_SENTENCE_LENGTH, topbestwords)

'''Reading from Pickle Objects:'''
dataframeFromDiskPos = pd.read_pickle(base_path_output + sentence_file_name_train_pos)
dataframeFromDiskNeg = pd.read_pickle(base_path_output + sentence_file_name_train_neg)
fileContentsPos = []
fileContentsNeg = []
fileNamesPos = []
fileNamesNeg = []
# print(pd.DataFrame(dataframeFromDiskPos).shape)
for a, b in zip(dataframeFromDiskPos[0], dataframeFromDiskPos[1]):
    fileContentsPos.append([a])
    fileNamesPos.append(b)
for a,b in zip(dataframeFromDiskNeg[0], dataframeFromDiskNeg[1]):
    fileContentsNeg.append([a])
    fileNamesNeg.append(b)

print(fileNamesNeg[0],fileNamesNeg[1])

'''Reading from Pickle Objects:'''
dataframeFromDiskPosVect = pd.read_pickle(base_path_output + 'savedPickleSentenceVectorPos.pickle')
dataframeFromDiskNegVect = pd.read_pickle(base_path_output + 'savedPickleSentenceVectorNeg.pickle')

print(pd.DataFrame(dataframeFromDiskNegVect).head())
print(pd.DataFrame(dataframeFromDiskPosVect).shape)

fileNamesNeg,vector_predict_sentence_classNeg, vector_predict_sentence_probNeg = [],[],[]
fileNamesPos,vector_predict_sentence_classPos, vector_predict_sentence_probPos = [],[],[]

for a,b,c in zip(dataframeFromDiskPosVect[0],dataframeFromDiskPosVect[1],dataframeFromDiskPosVect[2]):
    fileNamesPos.append(a)
    vector_predict_sentence_classPos.append(b)
    vector_predict_sentence_probPos.append(c)
for a,b,c in zip(dataframeFromDiskNegVect[0],dataframeFromDiskNegVect[1],dataframeFromDiskNegVect[2]):
    fileNamesNeg.append(a)
    vector_predict_sentence_classNeg.append(b)
    vector_predict_sentence_probNeg.append(c)

for a, b, c in zip(fileNamesNeg, vector_predict_sentence_probNeg, vector_predict_sentence_classNeg):
    print(a, end='\n')
    print(b, end='\n')
    print(c, end='\n')


print(fileNamesPos[0],fileNamesPos[1])
print(fileNamesNeg[0],fileNamesNeg[1])

posSentenceCount = 0
sent_prob = {}
print('Processing Positive Files', end='\n')
for x in range(0, len(fileContentsPos)):
    temp = [k for i in fileContentsPos[x] for j in i for k in j]
    sentence_split = str(temp[0]).split('.')
    # for i in sentence_split[0:len(sentence_split) - 1]:
    #     norm_text = normalizeText(i)
        # print(norm_text)
    # j = predictresp(model, finalSequenceUnitTokenizer, [norm_text])
    # for a, b in list(j):
    #     sentence_class = int(a)
    #     sentence_prob = float(b)
    #     # print('Hit')
    #     sent_prob[i] = sentence_prob
    posSentenceCount += len(sentence_split) - 1


negSentenceCount=0
print('Processing Negative Files',end='\n')
for x in range(0,len(fileContentsNeg)):
    temp=[k for i in fileContentsNeg[x] for j in i for k in j]
    sentence_split=str(temp[0]).split('.')
    # for i in sentence_split[0:len(sentence_split) - 1]:
    #     norm_text = normalizeText(i)
        # print(norm_text)
    # j = predictresp(model, finalSequenceUnitTokenizer, [norm_text])
    # for a, b in list(j):
    #     sentence_class = int(a)
    #     sentence_prob = float(b)
    #     sent_prob[i]=sentence_prob
    negSentenceCount+=len(sentence_split)-1

for i, j in sent_prob.items():
    print(str(i), float(j), end='\n')

print(posSentenceCount)
print(negSentenceCount)

predicted=np.zeros(len(fileNamesPos)+len(fileContentsNeg))

for i in range(0,len(predicted)):
    if(i<12500):
        # print(fileNamesPos[i])
        a=np.array(vector_predict_sentence_probPos[i])
    # print(a,end='\n')
        if len(a)!=0:
            temp=(np.sum(a)/len(a))
            predicted[i]=1 if temp>0.5 else 0
    else:
        # print(fileNamesNeg[i-12500])
        a = np.array(vector_predict_sentence_probNeg[i-12500])
        # print(a,end='\n')
        if len(a)!=0:
            temp=(np.sum(a)/len(a))
            predicted[i]=1 if temp>0.6 else 0
        # temp = (np.sum(a) / len(a))
        # predicted[i] = 1 if temp > 0.5 else 0
    # print(vector_predict_sentence_classPos[i],end='\n')
    # print(len(vector_predict_sentence_classPos[i]))

print(Counter(predicted[0:12500]))
print(Counter(predicted[12500:25000]))
#
# print(fileContentsPos[1:5])
#
# for i in range(0,len(allFileNamesTrain)):
#     print(str(allFileNamesTrain[i])+':'+str(predicted[i]))
#
#
#
#
# print(len(fileContentsPos))
# print(len(fileContentsNeg))
#
# dataframeFromDiskPos=pd.DataFrame(dataframeFromDiskPos)
# print(dataframeFromDiskPos.shape)

# vector_predict_sentence_class = []
# vector_predict_sentence_prob = []
# temp1 = []
# temp2 = []
# posSentenceCount = 0
# print('Processing Positive Files', end='\n')
# for x in range(0, len(fileContentsPos)):
#     temp = [k for i in fileContentsPos[x] for j in i for k in j]
#     sentence_split = str(temp[0]).split('.')
#     sys.stdout.write('\r')
#     sys.stdout.write("%d[%-100s]%d  %d%%" % (
#         x, '=' * int(round((x / len(fileContentsPos)) * 100, 0)), len(fileContentsPos),
#         int(round((x / len(fileContentsPos)) * 100, 0))))
#     sys.stdout.flush()
#     posSentenceCount += len(sentence_split) - 1
#     temp1 = []
#     temp2 = []
#     for i in sentence_split[0:len(sentence_split) - 1]:
#         norm_text = normalizeText(i)
#         # print(norm_text)
#         j = predictresp(model, finalSequenceUnitTokenizer, [norm_text])
#         for a, b in list(j):
#             sentence_class = int(a)
#             sentence_prob = float(b)
#         temp1.append(sentence_class)
#         temp2.append(sentence_prob)
#     vector_predict_sentence_class.append(temp1)
#     vector_predict_sentence_prob.append(temp2)
#     # print(vector_predict_sentence_class)
#     # print(vector_predict_sentence_prob)
#
# listToWrite = []
#
# for i, j, k in zip(fileNamesPos, vector_predict_sentence_class, vector_predict_sentence_prob):
#     listToWrite.append([i, j, k, 1])
# dataframeToDisk = pd.DataFrame(listToWrite)
# print(dataframeToDisk.shape)
# print(dataframeToDisk.head())
# # print(dataframeToDisk[1])
# dataframeToDisk.to_pickle(base_path_output + 'savedPickleSentenceVectorPosTest.pickle')
# print("Successfully Written Data to Disk (Pickle Object) !")
#
# print(posSentenceCount)
#
# print('Processing Negative Files', end='\n')
# vector_predict_sentence_class = []
# vector_predict_sentence_prob = []
# temp1 = []
# temp2 = []
# negSentenceCount = 0
# for x in range(0, len(fileNamesNeg)):
#     temp = [k for i in fileContentsNeg[x] for j in i for k in j]
#     sentence_split = str(temp[0]).split('.')
#     sys.stdout.write('\r')
#     sys.stdout.write("%d[%-100s]%d  %d%%" % (
#         x, '=' * int(round((x / len(fileContentsNeg)) * 100, 0)), len(fileContentsNeg),
#         int(round((x / len(fileContentsNeg)) * 100, 0))))
#     sys.stdout.flush()
#     negSentenceCount += len(sentence_split) - 1
#     temp1 = []
#     temp2 = []
#     for i in sentence_split[0:len(sentence_split) - 1]:
#         norm_text = normalizeText(i)
#         # print(norm_text)
#         j = predictresp(model, finalSequenceUnitTokenizer, [norm_text])
#         for a, b in list(j):
#             sentence_class = int(a)
#             sentence_prob = float(b)
#         temp1.append(sentence_class)
#         temp2.append(sentence_prob)
#     vector_predict_sentence_class.append(temp1)
#     vector_predict_sentence_prob.append(temp2)
#
# # print(fileNamesNeg[0],fileNamesNeg[1])
# listToWrite = []
# for i, j, k in zip(fileNamesNeg, vector_predict_sentence_class, vector_predict_sentence_prob):
#     listToWrite.append([i, j, k, 0])
# dataframeToDisk = pd.DataFrame(listToWrite)
# print(dataframeToDisk.shape)
# print(dataframeToDisk.head())
# dataframeToDisk.to_pickle(base_path_output + 'savedPickleSentenceVectorNegTest.pickle')
# print("Successfully Written Data to Disk (Pickle Object) !")
#
# print(negSentenceCount)
#
# N= posSentenceCount + negSentenceCount
# print(N)
'''
Implementing cost function:

N=sentences
lambda=any number > 0
K=no of groups

yi=yteta=from CNN predictions
yj=from CNN predictions


Kernel=exp(−kxi − xjk2 2) : pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
K = scip.exp(-pairwise_sq_dists / s**2)

lk=summation(yteta) : positive, else 0

1) Find number of sentences : positve, negative
2) We know groups: pos=12500 neg=12500
3) 

'''