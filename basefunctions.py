import os
import string
import sys
from collections import defaultdict, OrderedDict

import nltk as nlt
from nltk.corpus import stopwords

max_file_limit =10000
global_nature=""
# GroupLabel=dict()
wordVocab=defaultdict()

def refineSentence(fileContents):
    stopWords=stopwords.words('english').remove('not')
    stop = sorted(
        stopWords + list(string.punctuation) + ["i\\", "'m", "'s", "it\\", '...', "''", '``', 'br',
                                                                 's', '--'])
    tokens = nlt.tokenize.sent_tokenize(str(fileContents).lower())
    tokens = [w for w in tokens if w not in stop]
    return tokens

def refine(fileContents):
    stop = sorted(
        stopwords.words('english') + list(string.punctuation) + ["i\\", "'m", "'s", "it\\", '...', "''", '``', 'br',
                                                                 's', '--'])
    tokens = nlt.tokenize.word_tokenize(str(fileContents).lower())
    tokens = [w for w in tokens if w not in stop]
    global wordVocab
    wordVocab = OrderedDict(sorted(nlt.probability.FreqDist(tokens).items(), key=lambda t: t[1], reverse=True))

    return tokens

def readFiles(path, howManyFiles):
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
            abc=' '.join(refine(readContent))
            # global GroupLabel
            rating=int(current[current.find("_")+1:current.find(".txt")])
            GroupLabel.append(abc)
            fileRatings.append(rating)
            a += 1
    print(global_nature+ " Files Read: %d" % howManyFiles,end="\n")
    return str(fileContents),GroupLabel,fileNames,fileRatings

def wordFreqGenerator(words):
    d = defaultdict(int)
    for i in list(words):
        d[i] += 1
    return OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))


def writeHighFreqTermsToFile(location, fileToWrite, nature):
    global global_nature
    global_nature=nature

    fileContents,GroupLabel,fileNames,fileRatings = readFiles(location, max_file_limit)
    tokens = refine(fileContents)
    words = nlt.trigrams(tokens)
    vectDict = wordFreqGenerator(words)
    fileToWrite.write(nature + " Reviews Summary: \n\n")
    for i in vectDict.items():
        if i[1] > 1:
            fileToWrite.write(str(i) + "\n")
    temp = "There are " + str(vectDict.__len__()) + " " + nature + " Trigrams in Dictionary.\n\n"
    fileToWrite.write(temp)
    return tokens,vectDict,wordVocab,GroupLabel,fileNames,fileRatings