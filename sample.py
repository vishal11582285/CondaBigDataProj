# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:50:55 2017

@author: User
"""

max_file_limit=12500

from basefunctions import *

path = "data/aclImdb/train/pos"
fileContents=readFiles(path,max_file_limit)
tokens = refine(fileContents)
words=nlt.trigrams(tokens)
positiveVectDict=wordFreqGenerator(words)

path = "data/"
outputResult= open(path+"/"+"outputResult.txt", 'w',encoding="utf8")
outputResult.write("Positive Reviews Summary: \n\n")
for i in positiveVectDict.items():
    print(i)
    if (i[1] > 1):
        outputResult.write(str(i)+"\n")

temp="There are " + str(positiveVectDict.__len__()) +"Positive Trigrams in Dictionary.\n\n"
outputResult.write(temp)

outputResult.write("Negative Reviews Summary:\n\n")
path = "data/aclImdb/train/neg"
fileContents=readFiles(path,max_file_limit)
tokens = refine(fileContents)
words=nlt.trigrams(tokens)
negativeVectDict=wordFreqGenerator(words)
for i in negativeVectDict.items():
    print(i)
    if(i[1]>1):
        outputResult.write(str(i)+"\n")

temp="There are " + str(negativeVectDict.__len__()) +"Negative Trigrams in Dictionary.\n\n"
outputResult.write(temp)

outputResult.close()