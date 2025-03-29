import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Learning rate reduction if the loss plateaus
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Load Shakespeare dataset
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

# Define model with Dropout layers
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dropout(0.2))  # Dropout for regularization
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(len(characters), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Sampling function using Top-k sampling
def sample(preds, temperature=1.0, top_k=5):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    sorted_indices = np.argsort(preds)[-top_k:]
    sorted_probs = preds[sorted_indices] / np.sum(preds[sorted_indices])
    return np.random.choice(sorted_indices, p=sorted_probs)

# Load trained model
model = tf.keras.models.load_model('textgenerator2.keras')

def generate_text(length, temperature, top_k):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    
    for _ in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature, top_k)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    
    return generated

# User input for generation
length = int(input("Enter length of text to generate: "))
temperature = float(input("Enter temperature (0.2 to 1.0): "))
top_k = int(input("Enter top-k value for sampling (default 5): "))

print(generate_text(length, temperature, top_k))
