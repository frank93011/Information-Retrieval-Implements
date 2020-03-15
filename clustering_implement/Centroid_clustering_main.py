#%%
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pattern.en import lemma
from collections import Counter
import json
import os
import re
import math
import numpy as np
from numpy import inner
from numpy.linalg import norm
import heapq
import pickle
#%%
def process(file_path):
    with open(file_path,"r") as input:
        txt = input.read().lower()
        # txt = re.sub(r'[^\w\s]','',txt) #clear all non-word
        # txt = re.sub("^\d+\s|\s\d+\s|\s\d+$", "", txt)  #clear digits
        # word = txt.split()
        tokens = word_tokenize(txt)
        # remove all tokens that are not alphabetic
        word = [word for word in tokens if word.isalpha() or word.isalnum()]
        new = []
        stop = set(stopwords.words('english'))
        for w in word:
            if w not in stop:
                new.append(lemma(w))    #using pattern.en to do lemmatize
        return new

def cosine_similarity(x, y):
    cos_sim = inner(x, y)/(norm(x)*norm(y)) #using inner product to calculate cosine similarity
    return cos_sim

def tfIdf(folder_path, max_f = 5000, max_df = 0.8, min_df = 2):
    filePath = os.listdir(folder_path)
    document_frequency = Counter()  #initialization
    total_amount = len(filePath)    #total documents amount
    for i, filename in enumerate(filePath):
        wordList = process(folder_path+filename)
        df = Counter(set(wordList)) #using Counter to calculate term frequency
        dic = Counter(wordList)
        document_frequency += df

    idf = {}
    new_df = {}
    sortedterm=sorted(document_frequency.keys(), key=lambda x:x.lower())    #sort df by alphabetically
    for term in sortedterm:
        if(document_frequency[term] <= max_df * len(filePath) and document_frequency[term] >= min_df):
            idf[term] = math.log(total_amount/document_frequency[term], 10) #calculate idf
            new_df[term] = document_frequency[term] #re-sort the df dictionary

    tf_idf = []
    summary = np.array([0.0]*len(idf))
    for filename in (filePath):
        wordList = process(folder_path+filename)
        dic = Counter(wordList)
        document = dict.fromkeys(list(idf.keys()),0)    #initialize the term-frequncy dictionary and set all value to 0
        for term, idf_value in idf.items():
            document[term] = dic[term] * idf_value #calculating the tf-idf
        x = np.array(list(document.values()))
        x = x / (x**2).sum()**0.5   # transform to unit vector
        summary += x
        # tf_idf.append(x)
        tf_idf.append(dict(zip(np.array(list(document.keys())), x)))

    max_tfidf = dict(zip(tf_idf[0].keys(),summary))
    top_features = heapq.nlargest(max_f, max_tfidf, key = max_tfidf.get)
    new_tf_idf = []
    for i, doc in enumerate(tf_idf):
        temp = []
        for term in top_features:
            temp.append(doc[term])
        new_tf_idf.append(temp)
    
    return np.array(new_tf_idf)

def compute_centroid(dataset, data_points_index):
    size = len(data_points_index)
    dim = len(dataset[0])  # get the dimention of vector
    centroid = np.array([0.0]*dim)
    for idx in data_points_index:
        dim_data = dataset[idx]
        centroid += dim_data
    # for i in range(dim):
    centroid /= size
    return centroid

# triverse every node of the vectors and calculate similarity
def calculate_similarity(data):
    sim_table = []
    n = len(data)
    for i in range(n-1):
        for j in range(i+1, n):
            sim = cosine_similarity(data[i], data[j])
            sim_table.append((-1 * sim, [-1 * sim, [[i], [j]]])) # in order to build min heap so * -1
    return sim_table

def build_priority_queue(sim_table):
    # heapq.heapify(sim_table)
    heapq.heapify(sim_table) 
    heap = sim_table 
    return heap

def is_valid_heap(heap_node, old_clusters):
    pair_dist = heap_node[0]
    pair_data = heap_node[1]
    for old_cluster in old_clusters:
        if old_cluster in pair_data:
            return False
    return True

# re-trive all node with new set of clusters
def update_heap(heap, new_cluster, current_cluster):
    for c in current_cluster.values():
        new_heap = []
        sim = cosine_similarity(c['centroid'], new_cluster['centroid'])
        new_heap.append(sim)
        new_heap.append([new_cluster["elements"], c["elements"]])
        heapq.heappush(heap, (sim, new_heap))
        

def hac(data, k, sim_table):
    heap  = build_priority_queue(sim_table)
    old_cluster = []
    cluster = {}
    # initiate the single node clusters
    for i in range(len(data)):
        cluster[str([i])] = {}
        cluster[str([i])].setdefault("centroid", data[i])
        cluster[str([i])].setdefault("elements", [i])

    while len(cluster) > k:
        sim, max_item = heapq.heappop(heap) # get the first priority item
        pair_index = max_item[1] # get the two clusters' set
        if not is_valid_heap(max_item, old_cluster): #check if it contained already merged node
            continue

        new_cluster = {}
        new_cluster_elements = sum(pair_index, []) # get indexs in one list
        new_cluster_cendroid = compute_centroid(data, new_cluster_elements)
        new_cluster.setdefault("centroid", new_cluster_cendroid)
        new_cluster_elements.sort() # must kepp in the same order for comparison
        new_cluster.setdefault("elements", new_cluster_elements)
        for pair_item in pair_index:
            old_cluster.append(pair_item)
            del cluster[str(pair_item)]
        update_heap(heap, new_cluster, cluster) #get new heap with  new clusters
        cluster[str(new_cluster_elements)] = new_cluster
    # cluster.sort()
    return cluster
        
def pickle_dump(file_name, file):
    f = open(file_name, 'wb')
    # dump information to that file
    pickle.dump(file, f)
    # close the file
    f.close()

def txt_dump(cluster):
    category = len(cluster)
    output_text = ""
    for i in list(cluster.keys()):
        key = i[1:-1].split(', ')
        # print(key)
        for j in key:
            output_text += j + "\n"
        output_text +="\n"

    with open(str(category)+'.txt', 'w') as file:
        file.write(output_text[:-2])
    
    return output_text




#%%
tf_idf = tfIdf("IRTM/", 2000, 0.8, 2)
sim_table = calculate_similarity(tf_idf)
# %%
# pickle_dump("tfidf_simlarity_table", {"tf_idf":tf_idf, "sim_table":sim_table})
file = open("tfidf_simlarity_table", 'rb')
data = pickle.load(file)
tf_idf = data["tf_idf"]
sim_table = data["sim_table"]
#%%

cluster = hac(tf_idf, 8, sim_table)

# %%
k = 20
data = tf_idf

heap  = build_priority_queue(sim_table)
old_cluster = []
cluster = {}
# cluster = list(range(len(tf_idf)))
for i in range(len(data)):
    cluster[str([i])] = {}
    cluster[str([i])].setdefault("centroid", data[i])
    cluster[str([i])].setdefault("elements", [i])

while len(cluster) > k:
    sim, max_item = heapq.heappop(heap)
    pair_index = max_item[1]
    if not is_valid_heap(max_item, old_cluster):
        continue
    new_cluster = {}
    new_cluster_elements = sum(pair_index, [])
    new_cluster_cendroid = compute_centroid(data, new_cluster_elements)
    new_cluster.setdefault("centroid", new_cluster_cendroid)
    new_cluster_elements.sort()
    new_cluster.setdefault("elements", new_cluster_elements)
    for pair_item in pair_index:
        old_cluster.append(pair_item)
        del cluster[str(pair_item)]
    update_heap(heap, new_cluster, cluster)
    cluster[str(new_cluster_elements)] = new_cluster
# cluster.sort()

# %%
txt_dump(cluster)
# %%
