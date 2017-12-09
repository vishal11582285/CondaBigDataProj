from sample import load_model,readFromDisk,pickle_file_name_train,base_path_output,base_path_test,base_path_train
from basefunctions import normalizeText,writeHighFreqTermsToFile
from wordembedtensor import kerasTokenizerUnit
import os,sys
from nltk.corpus import stopwords
import nltk as nlt
import string
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

