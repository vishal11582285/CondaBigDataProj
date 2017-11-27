# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:50:55 2017

@author: Vishal Sonawane
"""

from collections import Counter

import numpy as np
import pandas as pd

from basefunctions import writeHighFreqTermsToFile
from wordembedtensor import kerasTokenizer,kerasTokenizerUnit,kerasTokenizerTest,CNNModel
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
base_path_train = "data/aclImdb/train/"
base_path_test = "data/aclImdb/test/"
pickle_file_name_train='storedDFforIMDBDatasetTrain.pickle'
pickle_file_name_test='storedDFforIMDBDatasetTest.pickle'

base_path_output = "data/"

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return string_data,index_data

def load_model():
    json_file = open(base_path_output + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(base_path_output+"model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])
    # score = loaded_model.evaluate(finalSequence, class_labels_norm, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    return loaded_model

def printDict(a):
    count=0
    for i in a.items():
        if(i[1]>=5):
            count+=1
            print(i,end="\n")
    print(count,end="\n\n")

def pad_sentences(sentences, padding_word=0):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def saveToDisk(allGroupKeys, allGroupValues, allFileNames, allFileRatings,fileName):
    listToWrite = []
    for i, j, k, l in zip(allGroupKeys, allGroupValues, allFileNames, allFileRatings):
        listToWrite.append([i, j, k, l])
    listToWrite = np.array(listToWrite)
    dataframeToDisk = pd.DataFrame(listToWrite)
    # print(dataframeToDisk.head())
    dataframeToDisk.to_pickle(base_path_output + fileName)
    return "Successfully Written Data to Disk (Pickle Object) !"

def readFromDisk(fileName):
    dataframeFromDisk = pd.read_pickle(base_path_output + fileName)
    allGroupKeys = []
    allGroupValues = []
    allFileNames = []
    allFileRatings = []
    for a, b, c, d in zip(dataframeFromDisk[0], dataframeFromDisk[1], dataframeFromDisk[2], dataframeFromDisk[3]):
        allGroupKeys.append(str(a))
        allGroupValues.append(int(b))
        allFileNames.append(str(c))
        allFileRatings.append(int(d))
    return allGroupKeys, allGroupValues, allFileNames, allFileRatings

def processInputTrain():
    outputResult = open(base_path_output + "/" + "outputResult.txt", 'w', encoding="utf8")
    positiveCorpus, positiveVectDict, wordFreqPos, GroupLabelPos, fileNamesPos, fileRatingsPos = writeHighFreqTermsToFile(
        base_path_train + "pos/", outputResult, "Positive")
    negativeCorpus, negativeVectDict, wordFreqNeg, GroupLabelNeg, fileNamesNeg, fileRatingsNeg = writeHighFreqTermsToFile(
        base_path_train + "neg/", outputResult, "Negative")
    outputResult.close()

    allGroupKeys = GroupLabelPos + GroupLabelNeg
    allGroupValues = fileRatingsPos + fileRatingsNeg
    allFileNames = fileNamesPos + fileNamesNeg
    allFileRatings = fileRatingsPos + fileRatingsNeg
    print(saveToDisk(allGroupKeys, allGroupValues, allFileNames, allFileRatings))

def processInputTrain():
    outputResult = open(base_path_output + "/" + "outputResult.txt", 'w', encoding="utf8")
    positiveCorpus, positiveVectDict, wordFreqPos, GroupLabelPos, fileNamesPos, fileRatingsPos = writeHighFreqTermsToFile(
        base_path_train + "pos/", outputResult, "Positive")
    negativeCorpus, negativeVectDict, wordFreqNeg, GroupLabelNeg, fileNamesNeg, fileRatingsNeg = writeHighFreqTermsToFile(
        base_path_train + "neg/", outputResult, "Negative")
    outputResult.close()

    allGroupKeys = GroupLabelPos + GroupLabelNeg
    allGroupValues = fileRatingsPos + fileRatingsNeg
    allFileNames = fileNamesPos + fileNamesNeg
    allFileRatings = fileRatingsPos + fileRatingsNeg
    print(saveToDisk(allGroupKeys, allGroupValues, allFileNames, allFileRatings,fileName=pickle_file_name_train))

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
    print(saveToDisk(allGroupKeys, allGroupValues, allFileNames, allFileRatings,fileName=pickle_file_name_test))


def getMAX_SENTENCE_LENGTH():
    MAX_SENTENCE_LENGTH = 0
    for i in allGroupKeysTrain:
        if (len(str([i]).split(' ')) > MAX_SENTENCE_LENGTH):
            MAX_SENTENCE_LENGTH = len(str([i]).split(' '))
    MAX_SENTENCE_LENGTH = min(MAX_SENTENCE_LENGTH, 1000)
    return MAX_SENTENCE_LENGTH

def saveModelToDisk(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(base_path_output + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(base_path_output + "model.h5")
    print("Saved model to disk")

def generateClassLabels(allGroupValues):
    class_labels = allGroupValues
    class_labels_norm = []
    custom_class_labels = []

    temp = np.zeros(len(allGroupValues))
    # print(temp)
    for i,j in zip(class_labels,range(len(allGroupValues))):
        # temp=list(np.zeros(1))
        if (i in [1, 2, 3, 4, 5]):
            temp[j] = 0
            custom_class_labels.append(0)
        if (i in [6, 7, 8, 9, 10]):
            temp[j] = 1
            custom_class_labels.append(1)
    temp=[int(i) for i in temp]
    return temp





'''Use this method if files are to be read and processed from disk.
Comment out if using saved Pickle object for faster operations'''
# processInput()

#Read from Pickle Object
print("Reading from Pickle Object Saved.",end="\n")
allGroupKeysTrain, allGroupValuesTrain, allFileNamesTrain, allFileRatingsTrain=readFromDisk(pickle_file_name_train)

MAX_SENTENCE_LENGTH=getMAX_SENTENCE_LENGTH()
# MAX_SENTENCE_LENGTH=300

topbestwords=1000
# finalSequence,dict_sequence=kerasTokenizer(GroupLabelPos,MAX_SENTENCE_LENGTH,topbestwords)
# for i in dict_sequence.items():
#     if(int(i[1])<=topbestwords):
#         print(i,end="\n")
#
# finalSequence,dict_sequence=kerasTokenizer(GroupLabelNeg,MAX_SENTENCE_LENGTH,topbestwords)
# for i in dict_sequence.items():
#     if(int(i[1])<=topbestwords):
#         print(i,end="\n")
finalSequence, dict_sequence = kerasTokenizer(allGroupKeysTrain, MAX_SENTENCE_LENGTH, topbestwords)

class_labels_norm=generateClassLabels(allGroupValuesTrain)

# print(class_labels_norm)

#Fit Model on Training Data. Comment out if model is saved to disk.
# model=CNNModel()

model=load_model()

# model.fit(finalSequence,class_labels_norm, epochs=10,batch_size=10)
# saveModelToDisk(model)

y_prob=np.zeros(len(allGroupKeysTrain))

finalSequenceUnitTokenizer = kerasTokenizerUnit(allGroupKeysTrain, MAX_SENTENCE_LENGTH, topbestwords)
for i,j in zip(allGroupKeysTrain,range(len(allGroupKeysTrain))):
    # finalSequenceUnit=kerasTokenizerUnit(allGroupKeys, MAX_SENTENCE_LENGTH, topbestwords,i)
    # print(finalSequenceUnit)
    # finalSequence = finalSequence.reshape(finalSequence.shape[0], finalSequence.shape[1])
    # finalSequemodence = finalSequence.reshape(finalSequence.shape[0], finalSequence.shape[1])
    this_sentence = list([i])
    sequences = finalSequenceUnitTokenizer.texts_to_sequences(this_sentence)
    finalSequenceUnit = pad_sequences(sequences, maxlen=MAX_SENTENCE_LENGTH, padding='pre')
    y_prob[j]=model.predict_classes(finalSequenceUnit,verbose=0)
    # print(y_prob[j])
    # y_prob[j]=[int(k) for i in y_prob[j] for k in i]

print("Predicted:")
print(Counter(y_prob))

print("Actual:")
print(Counter(class_labels_norm))

model=load_model()
score=model.evaluate(finalSequence,class_labels_norm,verbose=0)
print("The model performed with "+ str(round(score[1]*100,2))+" Accuracy.")


abc=[]
abc=[]
for i,j,k,l in zip(allFileNamesTrain,allFileRatingsTrain,class_labels_norm,y_prob):
    abc.append(list([i,j,k,l]))

dataFramePredicted=pd.DataFrame(abc,columns=list(["FileName","FileRating","TrueLabel","PredictedLabel"]))

summaryActualPredicted=open(base_path_output+"summaryActualPredicted.txt",'w')
# print(dataFramePredicted)
for i,j,k,l in zip(dataFramePredicted["FileName"],dataFramePredicted["FileRating"],dataFramePredicted["TrueLabel"],dataFramePredicted["PredictedLabel"]):
    summaryActualPredicted.write(str(i) + "\t"+str(j)+"\t"+str(k)+"\t"+str(l)+"\n")



'''Reading Training Data'''

#Read from Pickle Object

'''Use this method if Test set is to be read and processed from disk.
Comment out if using saved Pickle object for faster operations'''

# processInputTest()

print("Reading from Pickle Object Saved.",end="\n")
allGroupKeysTest, allGroupValuesTest, allFileNamesTest, allFileRatingsTest=readFromDisk(pickle_file_name_test)
finalSequenceTest, dict_sequenceTest = kerasTokenizerTest(allGroupKeysTrain,allGroupKeysTest, MAX_SENTENCE_LENGTH, topbestwords)

y_prob=np.zeros(len(allGroupKeysTest))

model=load_model()
# score=model.evaluate(finalSequenceTrain,class_labels_norm,verbose=0)

finalSequenceUnitTokenizer = kerasTokenizerUnit(allGroupKeysTrain, MAX_SENTENCE_LENGTH, topbestwords)
for i,j in zip(allGroupKeysTest,range(len(allGroupKeysTest))):
    # finalSequenceUnit=kerasTokenizerUnit(allGroupKeys, MAX_SENTENCE_LENGTH, topbestwords,i)
    # print(finalSequenceUnit)
    # finalSequence = finalSequence.reshape(finalSequence.shape[0], finalSequence.shape[1])
    # finalSequemodence = finalSequence.reshape(finalSequence.shape[0], finalSequence.shape[1])
    this_sentence = list([i])
    sequences = finalSequenceUnitTokenizer.texts_to_sequences(this_sentence)
    finalSequenceUnit = pad_sequences(sequences, maxlen=MAX_SENTENCE_LENGTH, padding='pre')
    y_prob[j]=model.predict_classes(finalSequenceUnit,verbose=0)
    # print(y_prob[j])
    # y_prob[j]=[int(k) for i in y_prob[j] for k in i]

class_labels_norm=generateClassLabels(allGroupValuesTest)

print("Predicted:")
print(Counter(y_prob))

print("Actual:")
print(Counter(class_labels_norm))

score=model.evaluate(finalSequenceTest,class_labels_norm,verbose=0)
print("The model performed with "+ str(round(score[1]*100,2))+" Accuracy.")
print("The model performed with "+ str(score[0])+" Loss.")

