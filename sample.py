# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:50:55 2017

@author: User
"""

from collections import defaultdict

def refine(temp):
    c=0
    skipWords=[",",".",")","("]
    for i in temp:
        if(any(n in i for n in skipWords)):  
            i=i[0:(len(i)-1)]
            temp[c]=i
        c+=1
    return temp
a=2

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

import os
import nltk as nlt

path="data/aclImdb/train/pos"
fileNames=os.listdir(os.path.abspath(path))

print(fileNames)
fileContents=[]

currentFile=fileNames[0]
openFile=open(path+"\\"+currentFile,'r')

fileContents.append(openFile.readline())

tokens=str(fileContents[0]).split(" ")
# print(tokens)
tokens = refine(tokens)
words=nlt.trigrams(tokens)
d = defaultdict(int)

for i in list(words):
    # print(i,end="\n")
    d[i] += 1

for i in set(zip(d.keys(),d.values())):
    print(i)

# for i in list(words):
#     print(i,end="\n")