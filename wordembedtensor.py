import numpy as np
from gensim.models import Word2Vec
from keras.layers import Dense, Flatten, LSTM, MaxPooling1D, Conv1D,Dropout
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD

vector_dim=0
# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 10

# Prepossessing parameters
sequence_length = 400
max_words = 10000

def kerasTokenizer(balanced_texts,max_sentence_length,topbestwords):
    global vector_dim
    vector_dim=max_sentence_length
    tokenizer = Tokenizer(num_words=topbestwords)
    tokenizer.fit_on_texts(balanced_texts)
    sequences = tokenizer.texts_to_sequences(balanced_texts)
    data = pad_sequences(sequences, maxlen=max_sentence_length,padding='pre')
    # print(tokenizer.word_index)
    return data,tokenizer.word_index

def RNNModel():
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=500))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def constructembedding(fileContents):
    model=Word2Vec(fileContents,min_count=1,size=vector_dim)
    embedding_matrix = np.zeros(vector_dim)
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return model,embedding_matrix

def CNNModel(embedding_matrix,max_limit):
    # saved_embeddings = tf.constant(embedding_matrix)
    # embedding_layer = Embedding(max_limit,
    #                             vector_dim,
    #                             weights=[embedding_matrix])
    # # sequence_input = Input(shape=(max_limit,vector_dim))
    model=Sequential()
    # embedded_sequences = embedding_layer(sequence_input)
    # print(model.output_shape)
    # model.add(Embedding(max_words,128,input_length=vector_dim))
    print(vector_dim)
    model.add(Conv1D(250,kernel_size=3, activation='relu',input_shape=(vector_dim,1)))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(250, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(250, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='sigmoid'))
    # sgd = SGD(lr=0.001, momentum=0.9, decay=0, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print("Output shape:",end="\n\n")
    print(model.output_shape)
    print(model.summary())
    return model
    # print(model.summary())
    # model.fit(embedding_matrix[1],epochs=2)