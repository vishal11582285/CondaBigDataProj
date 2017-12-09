from sample import load_model,readFromDisk,pickle_file_name_train
from basefunctions import normalizeText
from wordembedtensor import kerasTokenizerUnit
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import sqeuclidean


MAX_SENTENCE_LENGTH=1000
topbestwords=1000



def predictresp(model, finalSequenceUnitTokenizer,norm_text):
    sequences = finalSequenceUnitTokenizer.texts_to_sequences(norm_text)
    finalSequenceUnit = pad_sequences(sequences, maxlen=MAX_SENTENCE_LENGTH, padding='pre')
    model_resp_prob = model.predict(finalSequenceUnit, verbose=0)
    model_resp = model.predict_classes(finalSequenceUnit, verbose=0)
    print(model_resp_prob)
    print(model_resp)




model=load_model()
text=input('Enter a review:')
norm_text = [normalizeText(text)]
print(norm_text)

print("Reading from Pickle Object Saved.",end="\n")
allGroupKeysTrain, allGroupValuesTrain, allFileNamesTrain, allFileRatingsTrain=readFromDisk(pickle_file_name_train)
finalSequenceUnitTokenizer = kerasTokenizerUnit(allGroupKeysTrain, MAX_SENTENCE_LENGTH, topbestwords)

predictresp(model,finalSequenceUnitTokenizer,norm_text)




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

