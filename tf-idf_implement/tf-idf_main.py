#%%
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pattern.en import lemma
from collections import Counter
from sklearn.preprocessing import normalize
import json
import os
import re
import math
import numpy as np
from numpy import inner
from numpy.linalg import norm

#%%
def process(file_path):
    with open(file_path,"r") as input:
        txt = input.read().lower()
        txt = re.sub(r'[^\w\s]','',txt) #clear all non-word
        txt = re.sub("^\d+\s|\s\d+\s|\s\d+$", "", txt)  #clear digits
        word = txt.split()
        new = []
        stop = set(stopwords.words('english'))
        for w in word:
            if w not in stop:
                new.append(lemma(w))    #using pattern.en to do lemmatize
        return new

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]    #counting frequency of terms in each documents
    return dict(zip(wordlist,wordfreq))

def cosine_similarity(docX_path, docY_path):
    with open(docX_path) as fileX:  #read txt file as dictionary
        docX = json.loads(fileX.read())
    with open(docY_path) as fileY:
        docY = json.loads(fileY.read())
    x =  list(docX.values())
    y =  list(docY.values())
    cos_sim = inner(x, y)/(norm(x)*norm(y)) #using inner product to calculate cosine similarity
    return cos_sim
#%%
filePath = os.listdir("IRTM")
term_frequency = Counter()  #initialization
document_frequency = Counter()
total_amount = len(filePath)    #total documents amount
for i, filename in enumerate(filePath):
    # if i == 0:
    wordList = process("IRTM/"+filename)
    df = Counter(set(wordList)) #using Counter to calculate term frequency
    dic = Counter(wordList)
    term_frequency += dic
    document_frequency += df
#%%
idf = {}
new_df = {}
sortedterm=sorted(document_frequency.keys(), key=lambda x:x.lower())    #sort df by alphabetically
for term in sortedterm:
    idf[term] = math.log(total_amount/document_frequency[term], 10) #calculate idf
    new_df[term] = document_frequency[term] #re-sort the df dictionary

#%%
with open('dictionary.txt', 'w') as file:
     file.write(json.dumps(new_df)) #dump dictionary as txt
#%%
tf_idf = []
for filename in (filePath):
    wordList = process("IRTM/"+filename)
    dic = Counter(wordList)
    document = dict.fromkeys(list(idf.keys()),0)    #initialize the term-frequncy dictionary and set all value to 0
    for term, tf in dic.items():
        document[term] = tf * idf[term] #calculating the tf-idf
    x = np.array(list(document.values()))
    x = x / (x**2).sum()**0.5   # transform to unit vector
    tf_idf.append(dict(zip(np.array(list(document.keys())), x)))

#%%
for i in range(1, 3):
    with open('Doc' +str(i)+'.txt', 'w') as file:
        file.write(json.dumps(tf_idf[i-1])) #dump-tf-idf as txt file

cosine_similarity("Doc1.txt", 'Doc2.txt')


#%%
