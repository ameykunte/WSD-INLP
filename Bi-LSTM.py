import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.corpus import semcor
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


# Load and preprocess data
instances = semcor.instances()
word_senses = semcor.wordnet.senses()
context_sentences = []
target_senses = []
for instance in instances:
    context_sentences.append(word_tokenize(instance.context_sentence().lower()))
    target_senses.append(word_senses.index(instance.senses()[0]))
word_senses_dict = {sense: i for i, sense in enumerate(word_senses)}

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(context_sentences, target_senses, test_size=0.2, random_state=42)

# Tokenize data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad sequences to a fixed length
max_len = 100
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

# One-hot encode target labels
num_classes = len(word_senses_dict)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Define model architecture
inputs = Input(shape=(max_len,))
x = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128)(inputs)
x = Bidirectional(LSTM(units=128, return_sequences=True))(x)
x = Bidirectional(LSTM(units=64))(x)
outputs = Dense(units=num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=10)

# Evaluate the model on test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
