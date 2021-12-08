#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas
import re
import string

import tensorflow as tf
import numpy as np
import nltk

import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import random
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pandas.read_csv('../../texts.csv', names=["texts", "marks"])

nltk.data.path.append("../../nltk_data")
stop_words = stopwords.words("english")
stop_words.append(",")
stop_words.append(".")


# Model constants.
vocab_size = 20000
embedding_dim = 128
sequence_length = 1000

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    #remove stop words
    no_stop_words = ' ' + lowercase + ' '
    for each in stop_words:
        no_stop_words = tf.strings.regex_replace(no_stop_words, ' ' + each + ' ' , r" ")
    #print(no_stop_words)
    no_extra_space = tf.strings.regex_replace(no_stop_words, " +"," ")

    return tf.strings.regex_replace(
        no_extra_space, f"[{re.escape(string.punctuation)}]", ""
    )

def createModel():
    # A integer input for vocab indices.
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    return tf.keras.Model(inputs, predictions)
    
def endToEndModel(myModel):
    # A string input
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    # Turn strings into vocab indices
    indices = vectorize_layer(inputs)
    # Turn vocab indices into predictions
    outputs = myModel(indices)
    
    # Our end to end model
    return tf.keras.Model(inputs, outputs)

# Now that we have our custom standardization, we can instantiate our text
# vectorization layer. We are using this layer to normalize, split, and map
# strings to integers, so we set our 'output_mode' to 'int'.
# Note that we're using the default split function,
# and the custom standardization defined above.
# We also set an explicit maximum sequence length, since the CNNs later in our
# model won't support ragged sequences.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)

train_features, test_features, train_targets, test_targets = train_test_split(
        df['texts'], df['marks'],
        train_size=0.8,
        test_size=0.2,
        random_state=42,
        shuffle = True,
        stratify=df['marks']
    )

# train X & y
train_text_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(train_features.values, tf.string)
) 
train_cat_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(train_targets.values, tf.int64),

) 
# test X & y
test_text_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(test_features.values, tf.string)
) 
test_cat_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(test_targets.values, tf.int64),

)

# Now that the vocab layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary. You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.
vectorize_layer.adapt(train_features.to_numpy())
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices

def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return tf.squeeze(vectorize_layer(text))


# Vectorize the data.
train_ds_text = train_text_ds_raw.map(vectorize_text)
test_ds_text = test_text_ds_raw.map(vectorize_text)

train_ds = tf.data.Dataset.zip(
    (
            train_ds_text,
            train_cat_ds_raw
     )
)

test_ds = tf.data.Dataset.zip(
    (
            test_ds_text,
            test_cat_ds_raw
     )
)

batch_size = 16
AUTOTUNE = tf.data.experimental.AUTOTUNE
buffer_size= train_ds.cardinality().numpy()


train_ds = train_ds.shuffle(buffer_size=buffer_size)\
                   .batch(batch_size=batch_size,drop_remainder=True)\
                   .cache()\
                   .prefetch(AUTOTUNE)

test_ds = test_ds.shuffle(buffer_size=buffer_size)\
                   .batch(batch_size=batch_size,drop_remainder=True)\
                   .cache()\
                   .prefetch(AUTOTUNE)

myModelPath = "../../models/mymodel"
# myModel = tf.keras.models.load_model(myModelPath)
myModel = createModel()
# Compile the model with binary crossentropy loss and an adam optimizer.
myModel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 10

# Fit the model using the train and test datasets.
myModel.fit(train_ds, validation_data=test_ds, epochs=epochs)

myModel.save(myModelPath)
end_to_end_model = endToEndModel(myModel)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)
endToEndModelPath = "../../models/textmodel"
end_to_end_model.save(endToEndModelPath, save_format='tf')

