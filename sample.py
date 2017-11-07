# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:50:55 2017

@author: User
"""

from basefunctions import *

path = "data/aclImdb/train/pos"
fileContents=readFiles(path,howManyFiles=13000)
tokens = refine(fileContents)
words=nlt.trigrams(tokens)
positiveVectDict=wordFreqGenerator(words)

path = "data/"
outputResult= open(path+"/"+"outputResult.txt", 'w',encoding="utf8")
outputResult.write("Positive Reviews Summary: \n\n")
for i in positiveVectDict.items():
    print(i)
    outputResult.write(str(i))

temp="There are " + str(positiveVectDict.__len__()) +"Positive Trigrams in Dictionary.\n\n"
outputResult.write(temp)

outputResult.write("Negative Reviews Summary:\n\n")
path = "data/aclImdb/train/neg"
fileContents=readFiles(path,howManyFiles=12500)
tokens = refine(fileContents)
words=nlt.trigrams(tokens)
negativeVectDict=wordFreqGenerator(words)
for i in negativeVectDict.items():
    print(i)
    outputResult.write(str(i))

temp="There are " + str(negativeVectDict.__len__()) +"Negative Trigrams in Dictionary.\n\n"
outputResult.write(temp)

outputResult.close()