# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:50:55 2017

@author: Vishal Sonawane
"""

from collections import Counter
import numpy as np

from basefunctions import writeHighFreqTermsToFile
from wordembedtensor import kerasTokenizer, RNNModel, CNNModel

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
positiveCorpus,positiveVectDict,wordFreqPos,GroupLabelPos = writeHighFreqTermsToFile(base_path_train + "pos/", outputResult, "Positive")
# negativeCorpus,negativeVectDict,wordFreqNeg,GroupLabelNeg = writeHighFreqTermsToFile(base_path_train + "neg/", outputResult, "Negative")
outputResult.close()

#Generate Group Labels: (Document, IMDB_Score)
# for i in GroupLabel.items():
#     print(i,end="\n")

# print("Positive: ",end="\n")
# for i in positiveCorpus:
#     print(i)

# print("Negative: ",end="\n")
# printDict(wordFreqNeg)
listPos=[i for i in positiveVectDict.keys()]

# model,embedding_matrix=constructembedding(listPos)
# print(embedding_matrix[1:5])

#Print most common words
# print(model.wv.index2word[0:5],end="\n")
# print(model.wv.index2word[13108-100:13108],end="\n")

# http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
# print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])
# vocab_size=len(model.wv.vocab)
# print(vocab_size)
# print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3])
# wordList,wvRepr=convert_data_to_index(positiveCorpus,model.wv)
# train_model(model,embedding_matrix)
#
# print(wvRepr[:5],end="\n\n")
# print(wordList[:5],end="\n")
#
#

#
# wordembedResult = open(base_path_output + "/" + "wordembedResult.txt", 'w', encoding="utf8")
#
# # for i in positiveVectDict.keys():
# #     wordembedResult.write(str(i) + "\n" + str(model[i]))
# #     wordembedResult.write("\n")
#
# wordembedResult.close()
# ##Load and save model:
# model.save(root_path + "mymodel")
# model = gensim.models.Word2Vec.load(root_path + "mymodel")
MAX_SENTENCE_LENGTH=0
for i in GroupLabelPos.keys():
    # MAX_SENTENCE_LENGTH += len(str([i]).split(' '))
    if(len(str([i]).split(' '))>MAX_SENTENCE_LENGTH):
        MAX_SENTENCE_LENGTH = len(str([i]).split(' '))
# print(MAX_SENTENCE_LENGTH)
# MAX_SENTENCE_LENGTH=int(MAX_SENTENCE_LENGTH/len(GroupLabelPos))
# MAX_SENTENCE_LENGTH=500
MAX_SENTENCE_LENGTH=min(MAX_SENTENCE_LENGTH,500)
finalSequence,dict_sequence=kerasTokenizer([str(i) for i in GroupLabelPos.keys()],MAX_SENTENCE_LENGTH,topbestwords=5000)

class_labels=[i for i in GroupLabelPos.values()]

print(len(finalSequence),len(class_labels))

class_labels_norm=[]
for i in class_labels:
    temp=list(np.zeros(10))
    temp[i-1]=1
    class_labels_norm.append(temp)

print(Counter(class_labels))
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



finalSequence=finalSequence.reshape(finalSequence.shape[0],finalSequence.shape[1],1)
class_labels_norm=np.array(class_labels_norm)
# class_labels_norm.reshape(finalSequence.shape[0],class_labels_norm.shape[1],1)
model=CNNModel(finalSequence,len(GroupLabelPos))
print(model.summary())
# #
# # print(finalSequence[1])
# # print(class_labels_norm[1])
# #
model.fit(finalSequence,class_labels_norm, epochs=2)
print(model.summary())
y_prob = model.predict(finalSequence)
y_classes = y_prob.argmax(axis=-1)
print(Counter(y_classes))
# model.predict(finalSequence[:20],verbose=True)
print(Counter(class_labels))

# Make Predictions:
# y_prob = model.predict(finalSequence[:20])
# y_classes = y_prob.argmax(axis=-1)
# print(y_classes)