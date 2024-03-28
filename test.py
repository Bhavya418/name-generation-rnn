import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential

# Example names data
names = ['John', 'Jane', 'David', 'Alice', 'Bob', 'Emma']

# Convert names to lowercase and find vocabulary
names = [name.lower() for name in names]
chars = sorted(set(''.join(names)))
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Parameters
max_len = max([len(name) for name in names])
vocab_size = len(chars)

# Prepare training data
X = np.zeros((len(names), max_len), dtype=np.int32)
y = np.zeros((len(names), max_len, vocab_size), dtype=np.bool)
for i, name in enumerate(names):
    for t, char in enumerate(name):
        X[i, t] = char_to_index[char]
        if t < len(name) - 1:
            y[i, t, char_to_index[name[t + 1]]] = 1

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),  # Removed input_length
    LSTM(128, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
