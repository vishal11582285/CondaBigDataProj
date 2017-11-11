# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:50:55 2017

@author: Vishal Sonawane
"""

from collections import Counter

import numpy as np

from basefunctions import writeHighFreqTermsToFile
from wordembedtensor import kerasTokenizer, CNNModel

base_path_train = "data/aclImdb/train/"
base_path_output = "data/"

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return string_data,index_data

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


outputResult = open(base_path_output + "/" + "outputResult.txt", 'w', encoding="utf8")
positiveCorpus,positiveVectDict,wordFreqPos,GroupLabelPos,fileNamesPos,fileRatingsPos = writeHighFreqTermsToFile(base_path_train + "pos/", outputResult, "Positive")
negativeCorpus,negativeVectDict,wordFreqNeg,GroupLabelNeg,fileNamesNeg,fileRatingsNeg = writeHighFreqTermsToFile(base_path_train + "neg/", outputResult, "Negative")
outputResult.close()
MAX_SENTENCE_LENGTH=0

allGroupKeys=GroupLabelPos+GroupLabelNeg
allGroupValues=fileRatingsPos+fileRatingsNeg
allFileNames=fileNamesPos+fileNamesNeg

for i in allGroupKeys:
    if(len(str([i]).split(' '))>MAX_SENTENCE_LENGTH):
        MAX_SENTENCE_LENGTH = len(str([i]).split(' '))

MAX_SENTENCE_LENGTH=min(MAX_SENTENCE_LENGTH,1000)

# print(temp)
# print(len(temp))
finalSequence,dict_sequence=kerasTokenizer(allGroupKeys,MAX_SENTENCE_LENGTH,topbestwords=5000)

class_labels=allGroupValues
# print("Printing class labels:",end="\n")
# print(class_labels)
# print(allFileNames)
# print(len(finalSequence),len(class_labels))

class_labels_norm=[]
custom_class_labels=[]
for i in class_labels:
    temp=list(np.zeros(3))
    if(i in [1,2,3,4]):
        temp[0]=1
        custom_class_labels.append(0)
    if (i in [5,6,7]):
        temp[1] = 1
        custom_class_labels.append(1)
    if (i in [8,9,10]):
        temp[2] = 1
        custom_class_labels.append(2)
    class_labels_norm.append(temp)

# print(Counter(class_labels))
# print(class_labels)
# print(finalSequence[1])

#
# for i in model.wv.index2word[0:5]:
#     print(dict_sequence[i])

# for i in finalSequence[:5]:
#     print(i,end="\n")

#
# x= [str([i]).split(' ') for i in GroupLabelPos.keys()]
# normalizedSentences=pad_sentences(x)

# class_labels_norm=[i for i in class_labels]
# print(class_labels_norm)
# model=RNNModel()
# # model.fit(finalSequence,class_labels_norm, epochs=3)
# print(model.summary())

# class_labels_norm.reshape(finalSequence.shape[0],class_labels_norm.shape[1],1)
model=CNNModel()
# print(model.summary())
# #
# # print(finalSequence[1])
# # print(class_labels_norm[1])
# #
finalSequence=finalSequence.reshape(finalSequence.shape[0],finalSequence.shape[1],1)
class_labels_norm=np.array(class_labels_norm)
model.fit(finalSequence,class_labels_norm, epochs=2)
# print(model.summary())
y_prob = model.predict(finalSequence)
y_classes = y_prob.argmax(axis=-1)
print(Counter(y_classes))
# model.predict(finalSequence[:20],verbose=True)
print(Counter(custom_class_labels))

# Make Predictions:
# y_prob = model.predict(finalSequence[:20])
# y_classes = y_prob.argmax(axis=-1)
# print(y_classes)