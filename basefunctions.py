from collections import defaultdict, OrderedDict
import os,string
import nltk as nlt
from nltk.corpus import stopwords

# nlt.download_gui()

def refine(fileContents):
    # print(fileContents)
    # print(stopwords.words('english'))
    stop=sorted(stopwords.words('english') + list(string.punctuation) + ["i\\","'m","'s","it\\",'...', "''", '``', 'br', 's'])
    tokens = nlt.tokenize.word_tokenize(str(fileContents).lower())
    tokens = [w for w in tokens if not w in stop]
    return tokens

def readFiles(path,howManyFiles):
    fileNames = os.listdir(os.path.abspath(path))
    # howManyFiles=min(howManyFiles,12500)
    fileContents = []
    for current in fileNames[0:howManyFiles]:
        # print(current)
        currentFile = current
        with open(path + "//" + currentFile, 'r',encoding="utf8") as openFile:
            fileContents.append(openFile.readline())
    return str(fileContents)

def wordFreqGenerator(words):
    d = defaultdict(int)
    for i in list(words):
        d[i] += 1
    return OrderedDict(sorted(d.items(), key=lambda t: t[1],reverse=True))
