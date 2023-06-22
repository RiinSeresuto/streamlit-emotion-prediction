import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import json

# Reading file function
def open_file(path):
    with open(path, "r") as file:
        lines = file.readlines()
    
    return lines

# LOADING TRAINING DATASET
training_file = open_file("./dataset/training.csv")                 # Open the training.csv
training_data = [line.strip().split(',') for line in training_file] # Saving each line of training.csv as an array to an array
training_texts = [row[0] for row in training_data]                  # Training texts are separated and saved in an array
training_labels = np.array([int(row[1]) for row in training_data])  # Training labels are separated and saved in an array

# LOADING TESTING DATASET
testing_file = open_file("./dataset/test.csv")                      # Open the test.csv
testing_data = [line.strip().split(',') for line in testing_file]   # Saving each line of test.csv as an array to an array
testing_texts = [row[0] for row in testing_data]                    # Testing texts are separated and saved in an array
testing_labels = np.array([int(row[1]) for row in testing_data])    # Testing labels are separated and saved in an array

# TOKENIZING THE TEXTS
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(training_texts)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_texts)
training_padded = pad_sequences(training_sequences, padding="post", maxlen=100)

testing_sequences = tokenizer.texts_to_sequences(testing_texts)
testing_padded = pad_sequences(testing_sequences, padding="post", maxlen=100)

# Save the tokenizer as JSON
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as file:
    file.write(tokenizer_json)

# DEFINING AND COMPILING THE MODEL
num_classes = 6  # Number of classifications

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), verbose=2)

# Get the predicted labels for the testing data
predicted_labels = np.argmax(model.predict(testing_padded), axis=1)

# Print classification report
print(classification_report(testing_labels, predicted_labels))

# Print confusion matrix
print(confusion_matrix(testing_labels, predicted_labels))

#print("Saving model...")
#model.save("emotion-model.h5")
#print("Model saved.")