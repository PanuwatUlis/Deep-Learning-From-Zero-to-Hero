#Import Important Libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Get Data Set

path = "/Users/AdMiN/PycharmProjects/data_set/combined_data.csv"

dataset = pd.read_csv(path)
sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

#Create Subword Data Set
#By using tensorflow dataset
import tensorflow_datasets as tfds

vocab_size = 1000
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(sentences, vocab_size, max_subword_length=5)

#Check Tokenizer work appropiately
num = 5
print(sentences[num])
encoded = tokenizer.encode(sentences[num])
print(encoded)

#Replace sentences data with encode subwords
for i, sentence in enumerate(sentences):
    sentences[i] = tokenizer.encode(sentence)
#Check sentences are appropiately replace
print(sentences[5])

#Final pre-processing
#Parameters
max_length = 50
trunc_type = 'post'
padding_type ='post'

#Padding all sequences
sequences_padded = pad_sequences(sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#Seperate training set & testing set
training_size = int(len(sentences)*0.8)

training_sequences = sequences_padded[:training_size]
testing_sequences = sequences_padded[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

#Make labels to numpy array for input in neural network
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

#Create model using embedding
embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

#Train the model
num_epochs = 30
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(training_sequences, training_labels_final, epochs=num_epochs, validation_data=(testing_sequences, testing_labels_final))

#Plot the accuracy and loss
import matplotlib.pyplot as plt

def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val'+string])
    plt.show()
plot_graph(history, "accuracy")
plot_graph(history,"loss")

#Define function to predict the sentiment review
def predict_review(model, new_sentences, maxlen=max_length, show_padded_sequences=True):
    #Keep the original sentences so that we can keep using them later
    #Create an array to hold encode sequences
    new_sequences = []

    #Convert the new reviews to sequences
    for i, frvw in enumerate(new_sentences):
        new_sequences.append(tokenizer.encode(frvw))

    trunc_type = 'post'
    padding_type = 'post'

    #Padding all sequences for the new reviews
    new_reviews_padded = pad_sequences(new_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    classes = model.predict(new_reviews_padded)

    #The closer the class is to 1, the more positive the reviews is
    for x in range(len(new_sentences)):
        #We can see padded sequences if desired
        #Print the sequences
        if (show_padded_sequences):
            print(new_reviews_padded[x])
        #Print the review as text
        print(new_sentences[x])
        #Print its predited class
        print(classes[x])
        print('\n')

# Use the model to predict some reviews
"""fake_reviews = ["I love this phone",
                "Everything was cold",
                "Everything was hot exactly as I wanted",
                "Everything was green",
                "the host seated us immediately",
                "they gave us free chocolate cake",
                "we couldn't hear each other talk because of the shouting in the kitchen"
              ]

predict_review(model, fake_reviews)"""

#Define the function to train and show results of th models with deifferent layers
def fit_model_now(model, sentences):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(training_sequences, training_labels_final, epochs=num_epochs, validation_data=(testing_sequences, testing_labels_final))

    return history
def plot_results(history):
    plot_graph(history, "accuracy")
    plot_graph(history,"loss")
def fit_model_and_show_results(model, sentences):
    history = fit_model_now(model,sentences)
    plot_results(history)
    predict_review(model, sentences)

#Add the bidirectional LSTM
#Define model
model_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#fit_model_and_show_results(model_bidi_lstm, fake_reviews)

#Use Multiple bidirectional layers
model_multiple_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

#fit_model_and_show_results(model_multiple_bidi_lstm, fake_reviews)

#Compare the predictions for all models
my_reviews =["lovely", "dreadful", "stay away",
             "everything was hot exactly as I wanted",
             "everything was not exactly as I wanted",
             "they gave us free chocolate cake",
             "I've never eaten anything so spicy in my life, my throat burned for hours",
             "for a phone that is as expensive as this one I expect it to be much easier to use than this thing is",
             "we left there very full for a low price so I'd say you just can't go wrong at this place",
             "that place does not have quality meals and it isn't a good place to go for dinner",
             ]

print("===================================\n","Embeddings only:\n", "===================================",)
predict_review(model, my_reviews, show_padded_sequences=False)

print("===================================\n", "With a single bidirectional LSTM:\n",
      "===================================")
predict_review(model_bidi_lstm, my_reviews, show_padded_sequences=False)

print("===================================\n","With two bidirectional LSTMs:\n", "===================================")
predict_review(model_multiple_bidi_lstm, my_reviews, show_padded_sequences=False)