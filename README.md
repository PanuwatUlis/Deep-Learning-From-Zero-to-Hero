# Deep-Learning-From-Zero-to-Hero
Learning deep learning by using tensorflow. The purpose of this project is to gain more experience with deep learning and tensorflow. To prepare for Tensorflow Certificate Examination

#Natural Language Processing (NLP)
This one is a part of Intro to TensorFlow for Deep Learning from Udacity

Part#1: How to preprocessing data for NLP

>Create Tokenizer

>Tokenize sentences

>Make them as Sequences

>Make Sequences all the same length

===========================================================================

Part#2: Take data from kaggle and then follow the preprocessing process 

>Down load data from Kaggle

>use pandas to manipulate data to dataframe and then take data from text column to list in reviews variable

>follow preprocessing process like part#1

==========================================================================
Part#3: Word Embedding and Sentiment 

>Embedding the word to create vector of each word 

>build model with simple deep learning use embedding as 1st layer of deep learning

>After data pass throught embedding layer, transform the data one dimentional by using flatten or GlobalAveragePooling

>before the last layer we can use relu activation function for dense layer as many as you want

>Finallly the last layer only 1 unit to classify binary  classification with sigmoid function

===========================================================================

Part#4 Tweaking the model (Adjust Parameters)

>This part we gonna adjust some parameters, what we can adjust?

  - vocab_size 
  - embedding_dim 
  - max_length 
  - trunc_type
  - padding_type
  - Embed Layer
  - Dense Layer
  - Optimizer Model
  - etc.
  
In this case, follow the hand on project in Udacity, we adjust only vocab_size and max_length and see how accuracy we can improve the model
===============================================================================
