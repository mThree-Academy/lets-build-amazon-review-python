#!/usr/bin/env python
# coding: utf-8

# # Let's Build: A Machine Learning Algorithm that Determines Review Sentiment (Using Python 3)
# 
# Code for [Let's Build Blog Post](https://www.mthree.com/news/let-s-build-a-machine-learning-algorithm-that-determines-review-sentiment-using-python-3/41573/)
# 
# To be the first to know about our future 'Let's Build' walkthroughs, follow us on [LinkedIn](https://www.linkedin.com/company/mthree/), [Instagram](https://www.instagram.com/mthreeconsulting/), [Twitter](https://twitter.com/MthreeC) and [Facebook](https://www.facebook.com/mthreealumni/). 

# ## Step 1: Verify and load required packages

# In[ ]:


# load appropriate packages 
import nltk
import tensorflow as tf  
from tensorflow import keras
import numpy as np
import json
import sklearn
import matplotlib
print("Ready")


# # Step 2: Import the data and create the datasets
# 
# Dataset: [Kindle Store 5-core](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz)

# In[ ]:


import json 
import gzip
import pprint
dataset1 = list()
dataset2 = list()
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
# update path for local file as necessary
for l in parse("reviews_Kindle_Store_5.json.gz"):
    if int(l['overall']) == 1 or int(l['overall']) == 2:
        dataset1.append(l)
    if int(l['overall']) == 5:
        dataset2.append(l)

dataset1 = dataset1[:50000]
print("Number of negative reviews in the dataset: " + str(len(dataset1)))
print("Example of a negative review:")
pprint.pprint(dataset1[0])

dataset2 = dataset2[:50000]
print("\nNumber of positive reviews in the dataset: " + str(len(dataset2)))
print("Example of a positive review:")
pprint.pprint(dataset2[0])


# In[ ]:


# create a single dataset with all collected reviews

#dataset1: list of negative reviews 
#dataset2: list of positive reviews 
dataset = dataset1 + dataset2
print("Number of reviews in the dataset: " + str(len(dataset)))
print("\nFirst record in dataset:")
print(dataset[0])


# # Step 3: Split the Dataset

# In[ ]:


# Split the dataset into two different lists.
dataset_labels = list()
dataset_text = list()

num_reviews_positive  = 0
num_reviews_negative  = 0

# Iterate through the dataset
for row in dataset:
    if int(row['overall']) == 1 or int(row['overall']) == 2:
        dataset_labels.append(0) # negative: 0 
        num_reviews_negative = num_reviews_negative + 1   
    elif int(row['overall']) == 5:
        dataset_labels.append(1) # positive = 1  
        num_reviews_positive = num_reviews_positive + 1   
    else:
        continue
    dataset_text.append(row['reviewText'])
    
print("Number of negative reviews: " + str(num_reviews_negative))
print("Number of positive reviews: " + str(num_reviews_negative))

print("\nFirst record in dataset:")
# Display the text of a review
print(dataset_text[0]) #X
# Display the label assigned to the review
print(dataset_labels[0]) #Y

print("\nSecond record in dataset:")
print(dataset_text[1]) #X
print(dataset_labels[1]) #Y

# Machine learning 
# F(X) = Y : F has parameters: Estimate the parameters? 


# # Step 4: Convert text to integers

# In[ ]:


# Use the imdb dataset' index values to map our reviews dataset
imdb = keras.datasets.imdb

# Create a word_index dictionary where the key is a word and the value is an index of that token,
# using a prebuilt word_index dictionary from the imdb database
word_index = imdb.get_word_index()

# Add these key,value pairs to the word_index dictionary for padding
word_index["<PAD>"] = 0

# Because of the padding, we offset the original index values by 3
word_index = {k:(v + 3) for k,v in word_index.items()} 

print("Ready")

print("Index value for 'ready': " + str(word_index["ready"]))
print("Index value for 'enjoy': " + str(word_index["enjoy"]))
print("Index value for 'enjoying': " + str(word_index["enjoying"]))
print("Index value for 'john': " + str(word_index["john"]))

print("Number of items in index: " +str(len(word_index.keys())))
word_index["<PAD>"]
 
print("\nSample dictionary entries:")
print(dict(list(word_index.items())[0:10]))

# uncomment the following line to view the entire dictionary (which is very long!)
# print(word_index) 


# # Step 5: Tokenize the review text

# In[ ]:


# Add/update the NLTK word list
import nltk
nltk.download('punkt')


# In[ ]:


# Tokenize each review in the dataset 
# then map each token to an index and store the results in dataset_reviews_tokens_index
# use the word_tokenize from the nltk package to perform a quick tokenization.
# dataset_reviews_tokens_index is a list that will contain the list of token indexes of each reviews 

dataset_index = list()
for text in dataset_text:
    #tokenize each review
    tokens = nltk.word_tokenize(text)
    tokens_index = list()
    #iterate through indexes
    for token in tokens:
        #check if token exist in word_index
        if token in word_index:
            # append the index of the token to the list tokens_index, 
            # which contains the list of indexes of each review
            tokens_index.append(word_index[token])
        #if token doesn't exist in word_index then we ignore it.  
        else:
            continue
    # append the list of tokens_index to dataset_reviews_tokens_index
    dataset_index.append(tokens_index)

print("Original review: " + str(dataset_text[0]))
print("\nIndexed review: " + str(dataset_index[0]))
print(dataset_labels[0])


# ## Step 6: Truncate the review text

# In[ ]:


dataset_padded = keras.preprocessing.sequence.pad_sequences(dataset_index,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

print("This review has " + str(len(dataset_index[0])) + " words.")
print("Indexed review:  " + str(dataset_index[0]))

print("\nThis review has " + str(len(dataset_padded[0])) + " words.")
print("Padded review:  " + str(dataset_padded[0]))


# ## Step 7: Create the training and testing datasets

# In[ ]:


# Split data into training and testing data using the sklearn package
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(dataset_padded, dataset_labels, test_size=0.3)
print("Ready")
print(train_x[0])
print(test_x[0])


# ## Step 8: Training

# In[ ]:


# Training 
vocab_size = 88588 

# Create an empty sequential model 
model = keras.Sequential()

# Add input layer
model.add(keras.layers.Embedding(vocab_size, 16))

# Add a GlobalAveragePooling1D layer to the model
model.add(keras.layers.GlobalAveragePooling1D())

# Add a Dense layer to the model with 16 nodes
# tf.nn.relu Computes rectified linear: max(features, 0).
model.add(keras.layers.Dense(16, activation=tf.nn.relu))

# Add an output layer to the model with one single node 
# tf.nn.sigmoid Computes sigmoid of x element-wise Specifically, y = 1 / (1 + exp(-x)).
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# Display the summary of the model 
model.summary()
print("Ready")


# ## Step 9: Define the accuracy of the model

# In[ ]:


# Define the optimizer as adam 
# Define the loss function as a binary_crossentropy
# Define the performance metric as the accuracy of the model (rate of correct guesses)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['acc'])
print("Ready")


# ## Step 10: Define the validation set

# In[ ]:


# Define the size of the validation set
validation_size = 10000 

# extract the validation set from the training set 
x_val = train_x[:validation_size]

# extract the rest of the data as the training set 
partial_x_train = train_x[validation_size:]

# extract the labels of the validation set from the training set 
y_val = train_y[:validation_size]

# extract the rest of the labels as the training labels 
partial_y_train = train_y[validation_size:]

print("Ready")


# ## Step 11: Run the training model

# In[ ]:


# Run training on the training partial_x_train data 
# We include the validation data as input as well 

partial_y_train  = np.array(partial_y_train)
y_val = np.array(y_val)
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# ## Step 12: Test the model

# In[ ]:


# Apply model on testing data 
test_y = np.array(test_y)

results = model.evaluate(test_x, test_y)
#print(results)
#print(type(results))

from sklearn.metrics import confusion_matrix
a = model.predict_classes(test_x, verbose=0)
confusion_matrix(test_y, a)


# ## Step 13: Visualize the results

# In[ ]:


# Loss values for both training and validation

import matplotlib.pyplot as plt
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.clf() 
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# Accuracy of the training and validation epochs

plt.clf()  
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

