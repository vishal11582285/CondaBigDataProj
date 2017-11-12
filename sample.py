# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:50:55 2017

@author: Vishal Sonawane
"""

from collections import Counter

import numpy as np
import pandas as pd
from basefunctions import writeHighFreqTermsToFile
from wordembedtensor import kerasTokenizer, CNNModel
from keras.models import model_from_json

base_path_train = "data/aclImdb/train/"
# base_path_train = "data/aclImdb/data/train/"

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
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])
    score = loaded_model.evaluate(finalSequence, class_labels_norm, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

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

def saveToDisk(allGroupKeys, allGroupValues, allFileNames, allFileRatings):
    dataframeToDisk = np.empty((len(allGroupKeys), 4))
    listToWrite = []
    for i, j, k, l in zip(allGroupKeys, allGroupValues, allFileNames, allFileRatings):
        listToWrite.append([i, j, k, l])
    listToWrite = np.array(listToWrite)
    dataframeToDisk = pd.DataFrame(listToWrite)
    # print(dataframeToDisk.head())
    dataframeToDisk.to_pickle(base_path_output + 'storedDFforIMDBDataset.pickle')
    return "Successfully Written Data to Disk (Pickle Object) !"

def readFromDisk():
    dataframeFromDisk = pd.read_pickle(base_path_output + 'storedDFforIMDBDataset.pickle')
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

def processInput():
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

def getMAX_SENTENCE_LENGTH():
    MAX_SENTENCE_LENGTH = 0
    for i in allGroupKeys:
        if (len(str([i]).split(' ')) > MAX_SENTENCE_LENGTH):
            MAX_SENTENCE_LENGTH = len(str([i]).split(' '))
    MAX_SENTENCE_LENGTH = min(MAX_SENTENCE_LENGTH, 1000)
    return MAX_SENTENCE_LENGTH

def generateClassLabels(allGroupValues):
    class_labels = allGroupValues
    class_labels_norm = []
    custom_class_labels = []

    temp = np.zeros(len(allGroupValues))
    print(temp)
    for i,j in zip(class_labels,range(len(allGroupValues))):
        # temp=list(np.zeros(1))
        if (i in [1, 2, 3, 4, 5]):
            temp[j] = 0
            custom_class_labels.append(0)
        if (i in [6, 7, 8, 9, 10]):
            temp[j] = 1
            custom_class_labels.append(1)
        # if (i in [8,9,10]):
        #     temp[2] = 1
        #     custom_class_labels.append(2)
        # class_labels_norm.append(temp)

    # For binary only:
    # class_labels_norm = np.zeros(len(allGroupValues))
    # class_labels_norm[0:int(len(allGroupValues) / 2)] = 1
    # class_labels_norm[int(len(allGroupValues) / 2):len(allGroupValues)] = 0
    print(temp)
    return temp


# processInput()

allGroupKeys, allGroupValues, allFileNames, allFileRatings=readFromDisk()

MAX_SENTENCE_LENGTH=getMAX_SENTENCE_LENGTH()
# MAX_SENTENCE_LENGTH=300

topbestwords=50
# finalSequence,dict_sequence=kerasTokenizer(GroupLabelPos,MAX_SENTENCE_LENGTH,topbestwords)
# for i in dict_sequence.items():
#     if(int(i[1])<=topbestwords):
#         print(i,end="\n")
#
# finalSequence,dict_sequence=kerasTokenizer(GroupLabelNeg,MAX_SENTENCE_LENGTH,topbestwords)
# for i in dict_sequence.items():
#     if(int(i[1])<=topbestwords):
#         print(i,end="\n")
finalSequence, dict_sequence = kerasTokenizer(allGroupKeys, MAX_SENTENCE_LENGTH, topbestwords)

# for i in dict_sequence.items():
#     if (int(i[1]) <= topbestwords):
#         print(i, end="\n")

# for i in range(5):
#     print(GroupLabelPos[i] + "\t" + str(fileRatingsPos[i]),end="\n")
class_labels_norm=generateClassLabels(allGroupValues)

# print(class_labels_norm)

model=CNNModel()

#Working :
# finalSequence=finalSequence.reshape(finalSequence.shape[0],finalSequence.shape[1],1)
# class_labels_norm=np.array(class_labels_norm)

finalSequence=finalSequence.reshape(finalSequence.shape[0],finalSequence.shape[1])
# class_labels_norm=np.array(class_labels_norm)

# for i,j in zip(finalSequence,range(len(finalSequence))):
#     print(i,class_labels_norm[j],end="\n")

# model.fit(finalSequence,class_labels_norm, epochs=5, batch_size= 2)
model.fit(finalSequence,class_labels_norm, epochs=10,batch_size=10)
# model.save(base_path_output)
#
# plt.plot(hist.history['val_acc'])
# plt.plot(hist.history['acc'])
# plt.xlabel('cycle')
# plt.ylabel('accuracy')
# plt.show()

y_prob = model.predict_classes(finalSequence)
# y_prob=[i for i in y_prob]
y_prob=[int(j) for i in y_prob for j in i]
print(Counter(y_prob))
# y_classes = y_classes = keras.utils.np_util np_utils.probas_to_classes(y_prob)
# print(y_classes)
# model.predict(finalSequence[:20],verbose=True)
print(class_labels_norm)
print(Counter(class_labels_norm))

# serialize model to JSON
model_json = model.to_json()
with open(base_path_output+"model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(base_path_output+"model.h5")
print("Saved model to disk")

# later...

# load json and create model
