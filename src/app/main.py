#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas
import nltk

import numpy as np

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

nltk.data.path.append("./nltk_data")

stop_words = stopwords.words("english")
stop_words.append(",")
stop_words.append(".")

lem = WordNetLemmatizer()
likeness = [1, 0, 0, 1]

df = pandas.read_csv('texts.csv', names=["texts", "marks"])

#print(data)
#tokenized_text=sent_tokenize(data)
#tokenized_word=word_tokenize(data)
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
vocab = cv.fit(df['texts'])
text_counts= cv.transform(df['texts'])
#print(text_counts)
for x in range(30):
    X_train, X_test, y_train, y_test = train_test_split(
    text_counts, df['marks'], test_size=0.3, random_state=1)
    
    print(np.shape(X_test))
    # Model Generation Using Multinomial Naive Bayes
    clf = MultinomialNB().fit(X_train, y_train)
    
    
test_text = """For centuries, the study of human origins relied on fossils. Scientists wanted to know from which species Homo sapiens descended and to which we were most closely related. For answers, they could do little more than compare the size, shape and orientation of preserved bones and teeth — perhaps along with tools and other artifacts. That all changed by the 1980s. Scientists started looking at the DNA of people alive today as a way to understand our past. Then in the 1990s, genetics pulled off a feat straight out of science fiction: It decoded the DNA preserved in the fossils left by our ancient ancestors.
It was a pivotal change in the study of human evolution, notes John Hawks. He’s a paleoanthropologist. He works at the University of Wisconsin–Madison. That ancient DNA, he says, began revealing things no one had even thought to look for.

DNA started offering clues about our evolution in 1987.

That’s when researchers at the University of California, Berkeley analyzed DNA from people living around the world. They focused on what’s known as mitochondrial DNA. Children inherit this mtDNA from their mothers. It undergoes no genetic reshuffling. So mtDNA preserves a mother’s ancestry going back millennia.

African populations showed the greatest genetic diversity in mtDNA. And when the Berkeley team built a family tree using their genetic data, it contained two main branches. One held only African lines. The other contained lineages from all over the world, including Africa."""

test_counts = cv.transform([test_text])
print(np.shape(test_counts))
predicted= clf.predict(test_counts)
print(predicted)
