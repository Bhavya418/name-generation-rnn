
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# from keras.layers import Embedding, Dense, Concatenate
# import keras
# import keras.layers as L
# from keras.utils import to_categorical
# from keras.callbacks import LambdaCallback, ModelCheckpoint
# from keras.models import Sequential, load_model

# start_token = " "
# pad_token = "#"

# with open("facebook-names.txt") as f:
#     data = f.read().lower()[:-1].split("\n")
#     data = [start_token + name  for name in data]
# # print(data[:100])

# # print("Number of names:", len(data))
# # for name in data[::1000]:
# #     print(name)
# MAX_LENGTH = max(map(len, data))
# # print("Max length:", MAX_LENGTH)
# # plt.hist(list(map(len, data)), bins=25)
# # plt.show()

# tokens = list(set("".join(data)))
# tokens = list(sorted(tokens))
# # print(tokens)
# n_tokens = len(tokens)
# # print("n_tokens:", n_tokens)

# token_to_id = {token: i for i, token in enumerate(tokens)}
# token_to_id[pad_token] = len(token_to_id)
# # print(token_to_id)

# def to_num_matrix(names, ch_to_idx):
    
#     max_len = max(map(len, names))
#     names_idx = np.zeros((len(names), max_len), dtype='int32')
        
#     for i, name in enumerate(names):
#         for j, ch in enumerate(name):
#             names_idx[i][j] = ch_to_idx[ch]
        
#     return names_idx

# def acc_on_epoch_end(epoch, logs):
#     sys.stdout.flush()
#     print('\nValidation Accuracy: ' + str(compute_acc()*100) + ' %')
#     sys.stdout.flush()



# def generate_model_batches(names, batch_size=32, pad=0):
#     # no. of training examples
#     m = np.arange(len(names))
    
#     while True:
#         # get a shuffled index list
#         idx = np.random.permutation(m)
        
#         # start yeilding batches
#         for start in range(0, len(idx)-1, batch_size):
#             batch_idx = idx[start:start+batch_size]
#             batch_words = []
            
#             # take out the words and tags from 'batch_size' no. of training examples
#             for index in batch_idx:
#                 batch_words.append(names[index])
            
#             # input x
#             batch_x = to_num_matrix(batch_words,token_to_id)
            
#             # output labels 
#             batch_y_ohe = to_categorical(batch_x[:,1:], n_tokens)
#             return  batch_x[:,:-1], batch_y_ohe


# acc_callback = LambdaCallback(on_epoch_end=acc_on_epoch_end)
# model = Sequential()
# model.add(L.InputLayer([None], dtype='int32'))
# # embeddings layer
# model.add(L.Embedding(n_tokens, 16))

# # gru layer
# model.add(L.SimpleRNN(128, return_sequences=True, activation='tanh'))
# model.add(L.Dropout(0.5))
# model.add(L.BatchNormalization())
# # apply softmax
# model.add(L.TimeDistributed(L.Dense(n_tokens, activation='softmax')))
# # print(model.summary())
# BATCH_SIZE = 256

# # select optimizer
# optimizer = keras.optimizers.Adam()
# model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# # # train model
# hist = model.fit(generate_model_batches(data, batch_size=BATCH_SIZE), steps_per_epoch=len(data)/BATCH_SIZE,
#                           callbacks=[acc_callback], epochs=10)

# def sample_text(seed_phrase='a', max_length=max_name_len):
  
            
#     # input x to numerical index
#     text_x = to_num_matrix(list(list(seed_phrase)), ch_to_idx)
#     # index of current character
#     ch = ch_to_idx[seed_phrase]
    
#     # list of predicted character indices
#     indices = []
#     # counter
#     i = 0
    
#     #start generating
#     while ch != 0 and i<max_length:
#         # predict using the input character
#         pred = model.predict(text_x)[0]
        
#         # get the char with highest prob.
#         ch = np.random.choice(np.arange(len(vocab_tokens)) , p = pred.ravel())
        
#         if ch!=0:
#             indices.append(ch)
        
#         # feed the current char as the next input
#         text_x = to_num_matrix(list(list(idx_to_ch[ch])), ch_to_idx)
        
#         i = i+ 1
    
    
#     return ''.join([idx_to_ch[idx] for idx in indices])



import pandas as pd 
import numpy as np 
import os 
import string
import tensorflow as tf

file_name = "Names.txt"
with open (file_name, "r") as f:
    names = f.read().lower()[:-1].split("\n")

print("Number of names:", len(names))
print("Max length:", max(map(len, names)))

MAX_LENGTH = 10
names = [name for name in names if len(name) <= MAX_LENGTH]
print("Number of names:", len(names))

start_token = " "
pad_token = "#"
names = [start_token + name for name in names]
MAX_LENGTH+=1

tokens = sorted(set("".join(names+[pad_token])))
n_tokens = len(tokens)
print("n_tokens:", n_tokens)
print("tokens:", tokens)

token_to_id = {token: i for i, token in enumerate(tokens)}
id_to_token = {i: token for i, token in enumerate(tokens)}
print("token_to_id:", token_to_id)

def to_matrix(names, max_len=None, pad=token_to_id[pad_token], dtype=np.int32):

    max_len = max_len or max(map(len, names))
    names_ix = np.zeros([len(names), max_len], dtype) + pad

    for i in range(len(names)):
        name_ix = list(map(token_to_id.get, names[i]))
        names_ix[i, :len(name_ix)] = name_ix

    return names_ix

# print("".join(names[::5000]))
# print(to_matrix(names[::5000]))

X = to_matrix(names)
# print(X.shape)

X_train = np.zeros((X.shape[0],X.shape[1],n_tokens),np.int32)
y_train = np.zeros((X.shape[0],X.shape[1],n_tokens),np.int32)

for i, name in enumerate(X):
    
    for j in range(MAX_LENGTH-1):
        X_train[i,j,name[j]] = 1
        y_train[i,j,name[j+1]] = 1
    X_train[i,MAX_LENGTH-1,name[MAX_LENGTH-1]] = 1
    y_train[i,MAX_LENGTH-1,token_to_id[pad_token]] = 1
    
name_count = X.shape[0]
print("Number of training examples:", name_count)  
BATCH_SIZE = 64
steps_per_epoch = name_count // BATCH_SIZE

AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X,y_train))
    .shuffle(5000)
    .cache()
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

hidden_layers = 128
embedding_size = 16

from tensorflow.keras.layers import SimpleRNN, Dense, Embedding,Input
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(n_tokens, embedding_size))
model.add(SimpleRNN(hidden_layers, return_sequences=True,activation='tanh'))
model.add(SimpleRNN(hidden_layers, return_sequences=True,activation='tanh'))
model.add(Dense(n_tokens, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print(model.summary())


EPOCHS = 100
history = model.fit(train_dataset,steps_per_epoch=steps_per_epoch,epochs=EPOCHS)

def generateName(model, seed_phrase=" ", max_length=MAX_LENGTH):
    name = [seed_phrase]
    x = np.zeros((1,max_length),np.int32)
    x[0,0:len(seed_phrase)] = [token_to_id[token] for token in seed_phrase]
    for i in range(len(seed_phrase),max_length):  
        
        probs = list(model.predict(x)[0,i-1])
        
        probs = probs/np.sum(probs)
        
        index = np.random.choice(range(n_tokens),p=probs)
        
        if index == token_to_id[pad_token]:
            break
            
        x[0,i] = index
        
        name.append(tokens[index])
    
    return "".join(name)



seed_phrase = "a"
for _ in range(20):
    name = generateName(model,seed_phrase=seed_phrase)
    if name not in names:
        print(f"{name.lstrip()} (New Name)")
    else:
        print(name.lstrip())