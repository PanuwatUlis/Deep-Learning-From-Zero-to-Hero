#Import important library
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Write some sentences
sentences = [
    'My favorite food is ice cream',
    'do you like ice cream too?',
    'My dog likes ice cream!',
    "your favorite flavor of icecream is chocolate",
    "chocolate isn't good for dogs",
    "your dog, your cat, and your parrot prefer broccoli"
]

print(sentences)

#Create the tokenize
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')

#Tokenize the words
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

#Turn Sentences into Sequences
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

#Make The Sequences All The Same Length
padded = pad_sequences(sequences)
print("\nWord Index = ", word_index)
print("\nSequences = ", sequences)
print("\nPadded Sequences")
print(padded)

#Specify the max length for the padded sequences
padded_max = pad_sequences(sequences, maxlen=15)
print(padded_max)

#Put the padded at the end of sequences
padded_post = pad_sequences(sequences, maxlen=15, padding='post')
print(padded_post)

#Limit the lenght of sequences
padded_limit = pad_sequences(sequences, maxlen=3)
print(padded_limit)

