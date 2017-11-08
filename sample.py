# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:50:55 2017

@author: Vishal Sonawane
"""

from basefunctions import writeHighFreqTermsToFile
base_path_train = "data/aclImdb/train/"
base_path_output = "data/"

outputResult = open(base_path_output + "/" + "outputResult.txt", 'w', encoding="utf8")
positiveVectDict = writeHighFreqTermsToFile(base_path_train + "pos/", outputResult, "Positive")
negativeVectDict = writeHighFreqTermsToFile(base_path_train + "neg/", outputResult, "Negative")
outputResult.close()