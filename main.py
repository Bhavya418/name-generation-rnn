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

print("".join(names[::5000]))
print(to_matrix(names[::5000]))

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