import os
import string
from collections import defaultdict, OrderedDict

import nltk as nlt
from nltk.corpus import stopwords
from time import sleep
import sys
# nlt.download_gui()

max_file_limit = 1000
global_nature=""
def refine(fileContents):
    # print(fileContents)
    # print(stopwords.words('english'))
    stop = sorted(
        stopwords.words('english') + list(string.punctuation) + ["i\\", "'m", "'s", "it\\", '...', "''", '``', 'br',
                                                                 's', '--'])
    tokens = nlt.tokenize.word_tokenize(str(fileContents).lower())
    tokens = [w for w in tokens if w not in stop]
    return tokens


def readFiles(path, howManyFiles):
    fileNames = os.listdir(os.path.abspath(path))
    # howManyFiles=min(howManyFiles,12500)
    fileContents = []
    a = 1
    for current in fileNames[0:howManyFiles]:
        # print(current)
        currentFile = current
        with open(path + "//" + currentFile, 'r', encoding="utf8") as openFile:
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("%d[%-100s]%d  %d%%" % (a,'=' * int(round((a/howManyFiles)*100,0)),howManyFiles, int(round((a/howManyFiles)*100,0))))
            sys.stdout.flush()
        #     # print("Currently Reading File : " + currentFile + " .Poll Progress:" + "(" + str(a) + " of " + str(
        #     #     howManyFiles))
        # # print((a/howManyFiles)*100,end="\r")
            fileContents.append(openFile.readline())
            a += 1
    print(global_nature+ " Files Read: %d" % howManyFiles,end="\n")
    return str(fileContents)


def wordFreqGenerator(words):
    d = defaultdict(int)
    for i in list(words):
        d[i] += 1
    return OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))


def writeHighFreqTermsToFile(location, fileToWrite, nature):
    global global_nature
    global_nature=nature

    fileContents = readFiles(location, max_file_limit)
    tokens = refine(fileContents)
    words = nlt.trigrams(tokens)
    vectDict = wordFreqGenerator(words)
    fileToWrite.write(nature + " Reviews Summary: \n\n")
    for i in vectDict.items():
        # print(i)
        if i[1] > 1:
            fileToWrite.write(str(i) + "\n")
    temp = "There are " + str(vectDict.__len__()) + " " + nature + " Trigrams in Dictionary.\n\n"
    fileToWrite.write(temp)
    return vectDict
