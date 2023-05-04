import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, Dropout
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Load data
train_sentences = load_train_sentences()
train_labels = load_train_labels()
test_sentences = load_test_sentences()
test_labels = load_test_labels()

# Tokenize sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

# Padding
train_sequences = pad_sequences(train_sequences)
test_sequences = pad_sequences(test_sequences)

# Define model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=train_sequences.shape[1]))
model.add(SimpleRNN(100))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(train_labels)), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(train_sequences, keras.utils.to_categorical(train_labels), batch_size=32, epochs=10, validation_split=0.1)

# Evaluate model
loss, accuracy = model.evaluate(test_sequences, keras.utils.to_categorical(test_labels))
print('Test accuracy:', accuracy)

# Predict labels for test data
test_pred = np.argmax(model.predict(test_sequences), axis=1)

# Print classification report
precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, test_pred, average='weighted')


# Calculate and print precision, recall, and F1 score
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)

# Calculate and print loss and val_loss
loss, val_loss = model.evaluate(test_sequences, keras.utils.to_categorical(test_labels), verbose=0)
print('Loss:', loss)
print('Val_loss:', val_loss)
