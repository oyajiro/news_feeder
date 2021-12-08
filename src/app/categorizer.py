#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import re
import string
import nltk
import sys

from nltk.corpus import stopwords
from sys import exit

arguments = sys.argv
if (len(arguments) != 2):
    exit('one argument with text to categorise must be provided')
text = arguments[1]

endToEndModelPath = "../../models/textmodel"

nltk.data.path.append("../../nltk_data")
stop_words = stopwords.words("english")

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

model = tf.keras.models.load_model(endToEndModelPath)
predictions = model.predict([text])
print(np.argmax(predictions[0]))