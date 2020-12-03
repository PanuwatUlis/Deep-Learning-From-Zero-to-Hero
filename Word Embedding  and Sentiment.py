#Import TensorFlow and Related Functions
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

#Get the corpus of text
path = "/Users/AdMiN/PycharmProjects/data_set/combined_data.csv"

dataset = pd.read_csv(path)

sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

#Separate out the sentences and labels to train and test set
training_size = int(len(sentences)*0.8)

training_sentences = sentences[:training_size]
test_sentences = sentences[training_size:]
training_labels = labels[:training_size]
test_labels = labels[training_size:]

#Make labels to numpy array to use for the network later
training_labels_final = np.array(training_labels)
test_labels_final = np.array(test_labels)

#Tokenize the data set
vocab_size = 1000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

#Tokenize Training Set
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#Tokenize Test Set
testing_sequences = tokenizer.texts_to_sequences(test_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#Train Basic Sentiment Model with Embedding
#Build basic sentiment network
#Embedding layer is the first one
model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(6, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, test_labels_final))

#Predict the sentiment in new reviews
fake_reviews = ['I love this phone', 'I hate spaghetti',
                'Everything was cold',
                'Everything was hot exactly as I wanted',
                'Everything was green',
                'the host seated us immediately',
                'they gave us free chocolate cake',
                'not sure about the wilted flowers on the table',
                'only works when I stand on tippy toes',
                'does not work when I stand on my head']

print(fake_reviews)

#Create sequences
sample_sequences = tokenizer.texts_to_sequences(fake_reviews)
fake_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)

print('\nHot OFF THE PRESS! HERE ARE SOME NEWLY MINTED, ABSOLUTELY GENUINE REVIEWS! \n')

classes = model.predict(fake_padded)

#The closer the class is to 1, the more positive review is deemed to be
for x in range(len(fake_reviews)):
    print(fake_reviews[x])
    print(classes[x])
    print('\n')

