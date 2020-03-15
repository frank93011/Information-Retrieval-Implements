#%%
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pattern.en import lemma
from collections import Counter
import os
import re
import math
import numpy as np
import pandas as pd

#%%
def process(doc):
    txt = doc.lower()
    txt = re.sub(r'[^\w\s]',' ',txt) #clear all non-word
    txt = re.sub('[0-9]', '', txt)
    word = txt.split()
    new = []
    stop = set(stopwords.words('english'))
    for w in word:
        if w not in stop:
            new.append(lemma(w))    #using pattern.en to do lemmatize
    return new

def MI(f_x, f_y, f_xy):
    return math.log(f_xy/ f_x * f_y)

def chi2(observed, t_present, c_on_topic, N):
    expected = N * t_present/N * c_on_topic/N
    return ((observed - expected)**2 / expected)

#%%
with open('d:/Users/bioha/Desktop/proj_school/senior/first/IR/PA-3/training.txt','r') as fp:
     all_lines = fp.readlines()

#%%
data = [] # record count of each terms in different category
doc_feature = {}
doc_feature_sum = []
docs_per_category = [] # count how many doc in a category
docs_counter = Counter() # terms count for all documents
for i in range(13):
    tagged_doc = ""
    doc_feature[i] = []
    train_set = all_lines[i].split()[1:]
    docs_per_category.append(len(train_set))
    print(train_set)
    for j in train_set:
        with open('d:/Users/bioha/Desktop/proj_school/senior/first/IR/IRTM/'+str(j)+'.txt','r') as fp:
            doc = fp.read()
            tagged_doc += doc
            doc_feature[i] += process(doc)
            doc_feature_sum += process(doc)
        # tagged_doc.append(doc)
    data.append(Counter(process(tagged_doc)))
    docs_counter += data[i]

#%%
top_features = []
selected_data = []
doc_cnt = Counter(doc_feature_sum)
for i in range(13):
    dic_chi = {}
    total_terms = len(data[i])
    for word, cnt in data[i].items():
        observed = 0
        cat_cnt = Counter(doc_feature[i])
        observed = cat_cnt[word]
        print(word, observed, doc_cnt[word], docs_per_category[i], sum(docs_per_category))
        try:
            dic_chi[word]= chi2(observed, doc_cnt[word], docs_per_category[i], sum(docs_per_category))
        except:
            pass
    dic_chi = pd.DataFrame(list(dic_chi.items())).sort_values(by = 1, ascending = False)
    top_features = (list(dic_chi.head(500)[0]))
    new_dic = {}
    for word, cnt in data[i].items():
        if(word in top_features):
            new_dic[word] = cnt
    data[i] = new_dic


# %%
total_docs = sum(docs_per_category)
feature_log = []
prior_log = []
for i in range(13):
    prob = {}
    prior_log.append(docs_per_category[i]/total_docs)
    total_tokens = sum(data[i].values())
    for word, cnt in data[i].items():
        prob[word] = (cnt+1)/(len(data[i]) + len(docs_counter))
    feature_log.append(prob)

# %%
out = pd.read_csv("d:/Users/bioha/Desktop/proj_school/senior/first/IR/PA-3/hw3_sam.csv")
test = list(out['Id'])

max_category = []
for i, id in enumerate(test):
    with open('d:/Users/bioha/Desktop/proj_school/senior/first/IR/IRTM/'+str(id)+'.txt','r') as fp:
        doc = process(fp.read())
    # doc = process('d:/Users/bioha/Desktop/proj_school/senior/first/IR/IRTM/'+str(id)+'.txt')
    score = []
    for j in range(13):
        score_temp = math.log(prior_log[j],10)
        total_tokens = sum(data[j].values())
        for word in doc:
            if(word in list(feature_log[j].keys())):
                score_temp += math.log(feature_log[j][word],10)
            else:
                score_temp += math.log(1/(len(data[j]) + len(docs_counter)), 10)
        score.append(score_temp)
    max_category.append(np.argmax(score)+1)
    
# %%
out['Value'] = max_category
out.to_csv('output.csv', index=False)
# %%
