import numpy as np
from keras.preprocessing.sequence import pad_sequences

from basefunctions import normalizeText
from sample import load_model, readFromDisk, pickle_file_name_train
from wordembedtensor import kerasTokenizerUnit

MAX_SENTENCE_LENGTH=1000
topbestwords=1000

model=load_model()

text=input('Enter a review:')
norm_text = [normalizeText(text)]
print(norm_text)

# print("Reading from Pickle Object Saved.",end="\n")
allGroupKeysTrain, allGroupValuesTrain, allFileNamesTrain, allFileRatingsTrain=readFromDisk(pickle_file_name_train)
finalSequenceUnitTokenizer = kerasTokenizerUnit(allGroupKeysTrain, MAX_SENTENCE_LENGTH, topbestwords)

sequences = finalSequenceUnitTokenizer.texts_to_sequences(norm_text)
finalSequenceUnit = pad_sequences(sequences, maxlen=MAX_SENTENCE_LENGTH, padding='pre')
model_resp = np.zeros(1)
model_resp_prob = model.predict(finalSequenceUnit, verbose=0)
model_resp = model.predict_classes(finalSequenceUnit, verbose=0)

print('Predicted probability : ', (model_resp_prob))
print('Predicted class label : ', model_resp)
