#
# from time import sleep
# import sys
# from nltk.corpus import stopwords
# from keras.datasets import imdb
# from keras.models import Sequential
# from keras.layers import  Embedding
# # for i in range(51):
# #     sys.stdout.write('\r')
# #     # the exact output you're looking for:
# #     sys.stdout.write("[%-50s] %d%%" % ('='*i, 2*i))
# #     sys.stdout.flush()
# #     sleep(0.25)
#
#
# def kerasimdb():
#     (X_train, y_train), (X_test,y_test) = imdb.load_data(nb_words=300)
#     print(X_train[1:10])
#
# kerasimdb()
#
# def buildmodel(embedding):
#     model=Sequential()
#     model.add(Embedding(weights=[embedding],name="MyEmbedding"))
#
#
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []
# for train, test in kfold.split(X, Y):
#   # create model
# 	model = Sequential()
# 	model.add(Dense(12, input_dim=8, activation='relu'))
# 	model.add(Dense(8, activation='relu'))
# 	model.add(Dense(1, activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	# Fit the model
# 	model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
# 	# evaluate the model
# 	scores = model.evaluate(X[test], Y[test], verbose=0)
# 	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# 	cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
#
#
# # print("First lengths: ",len(GroupLabelPos),len(GroupLabelNeg))
# # for i in GroupLabelPos.items():
# #     print(i,end="\n")
# #
# # print("First lengths: ",len(GroupLabelPos),len(GroupLabelNeg))
# # for i in GroupLabelNeg.items():
# #     print(i,end="\n")
# #Generate Group Labels: (Document, IMDB_Score)
# # for i in wordFreqPos.items():
# #     print(i,end="\n")
#
# # print("Positive: ",end="\n")
# # for i in positiveCorpus:
# #     print(i)
#
# # print("Negative: ",end="\n")
# # printDict(wordFreqNeg)
# # listPos=[i for i in positiveVectDict.keys()]
#
# # model,embedding_matrix=constructembedding(listPos)
# # print(embedding_matrix[1:5])
#
# #Print most common words
# # print(model.wv.index2word[0:5],end="\n")
# # print(model.wv.index2word[13108-100:13108],end="\n")
#
# # http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
# # print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])
# # vocab_size=len(model.wv.vocab)
# # print(vocab_size)
# # print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3])
# # wordList,wvRepr=convert_data_to_index(positiveCorpus,model.wv)
# # train_model(model,embedding_matrix)
# #
# # print(wvRepr[:5],end="\n\n")
# # print(wordList[:5],end="\n")
# #
# # wordembedResult = open(base_path_output + "/" + "wordembedResult.txt", 'w', encoding="utf8")
# #
# # # for i in positiveVectDict.keys():
# # #     wordembedResult.write(str(i) + "\n" + str(model[i]))
# # #     wordembedResult.write("\n")
# #
# # wordembedResult.close()
# # ##Load and save model:
# # model.save(root_path + "mymodel")
# # model = gensim.models.Word2Vec.load(root_path + "mymodel")
# # print(temp)
# # print(len(temp))
# # print("Printing class labels:",end="\n")
# # print(class_labels)
# # print(allFileNames)
# # print(len(finalSequence),len(class_labels))
#
# # print(Counter(class_labels))
# # print(class_labels)
# # print(finalSequence[1])
#
# #
# # for i in model.wv.index2word[0:5]:
# #     print(dict_sequence[i])
#
# # for i in finalSequence[:5]:
# #     print(i,end="\n")
#
# #
# # x= [str([i]).split(' ') for i in GroupLabelPos.keys()]
# # normalizedSentences=pad_sentences(x)
#
# # class_labels_norm=[i for i in class_labels]
# # print(class_labels_norm)
# # model=RNNModel()
# # # model.fit(finalSequence,class_labels_norm, epochs=3)
# # print(model.summary())
#
# # class_labels_norm.reshape(finalSequence.shape[0],class_labels_norm.shape[1],1)



#Working :
# finalSequence=finalSequence.reshape(finalSequence.shape[0],finalSequence.shape[1],1)
# class_labels_norm=np.array(class_labels_norm)


# finalSequence=finalSequence.reshape(finalSequence.shape[0],finalSequence.shape[1])
# class_labels_norm=np.array(class_labels_norm)

# for i,j in zip(finalSequence,range(len(finalSequence))):
#     print(i,class_labels_norm[j],end="\n")

# model.fit(finalSequence,class_labels_norm, epochs=5, batch_size= 2)

# This one:
# This one:
# model.fit(finalSequence,class_labels_norm, epochs=1,batch_size=10)
# saveModelToDisk(model)


# for i in dict_sequence.items():
#     if (int(i[1]) <= topbestwords):
#         print(i, end="\n")

# for i in range(5):
#     print(GroupLabelPos[i] + "\t" + str(fileRatingsPos[i]),end="\n")



# MAX_SENTENCE_LENGTH=300


# finalSequence,dict_sequence=kerasTokenizer(GroupLabelPos,MAX_SENTENCE_LENGTH,topbestwords)
# for i in dict_sequence.items():
#     if(int(i[1])<=topbestwords):
#         print(i,end="\n")
#
# finalSequence,dict_sequence=kerasTokenizer(GroupLabelNeg,MAX_SENTENCE_LENGTH,topbestwords)
# for i in dict_sequence.items():
#     if(int(i[1])<=topbestwords):
#         print(i,end="\n")

# print(class_labels_norm)
