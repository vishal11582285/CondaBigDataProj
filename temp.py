
from time import sleep
import sys
from nltk.corpus import stopwords
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import  Embedding
# for i in range(51):
#     sys.stdout.write('\r')
#     # the exact output you're looking for:
#     sys.stdout.write("[%-50s] %d%%" % ('='*i, 2*i))
#     sys.stdout.flush()
#     sleep(0.25)


def kerasimdb():
    (X_train, y_train), (X_test,y_test) = imdb.load_data(nb_words=300)
    print(X_train[1:10])

kerasimdb()

def buildmodel(embedding):
    model=Sequential()
    model.add(Embedding(weights=[embedding],name="MyEmbedding"))


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
  # create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
	# evaluate the model
	scores = model.evaluate(X[test], Y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))