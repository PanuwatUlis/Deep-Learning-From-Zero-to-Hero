#Import Tokenizer and pad_sequences
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#import numpy and pandas
import numpy as np
import pandas as pd

#Get the corpus of text
path = "/Users/AdMiN/PycharmProjects/data_set/combined_data.csv"

dataset = pd.read_csv(path)

print(dataset.head())

#Get the review from csv file
reviews = dataset['text'].tolist()

#Tokenize the text
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)

#Generate Sequences for reviews
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, padding="post")

print(padded_sequences.shape)
print(reviews[0])
print(padded_sequences[0])





