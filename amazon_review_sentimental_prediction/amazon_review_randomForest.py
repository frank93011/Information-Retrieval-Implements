#%%
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import re
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from sklearn import utils
from sklearn.feature_extraction.text import CountVectorizer

train_size = 50000
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# news data 

#%%
train = pd.read_csv('train.csv',nrows=50000)
test = pd.read_csv('test.csv',nrows=5000)

#%%
print(train.head(10))

#%%
r1 = '[0-9’!１２３４５６７８９０"#$%&\'()（）*+,-/:;<=>?@，。?★、…【】《》＊;「」？“”‘’！：[\\]^_`{|}~‧]+'
train[train.columns[2]] = train[train.columns[2]].apply(lambda x: re.sub(r1,'',x))

#%%
# ============== pos tagging ==============
train_split = []
test_split = []

for i in range(train_size):
    row = train.loc[i]
    temp = nltk.pos_tag(word_tokenize(row[2]))
    nounVerb = []
    required_type = ['JJ', 'JJR', 'JJS']
    for i in temp:
        if i[1] in required_type:
            nounVerb.append(lemmatizer.lemmatize(i[0]))
    train_split.append(nounVerb)

# test part
for i in range(5000):
    row = test.loc[i]
    temp = nltk.pos_tag(word_tokenize(row[2]))
    nounVerb = []
    required_type = ['JJ', 'JJR', 'JJS']
    for i in temp:
        if i[1] in required_type:
            nounVerb.append(lemmatizer.lemmatize(i[0]))
    test_split.append(nounVerb)

#%%
D_train = []
for i in range(len(train_split)):
	str1 = ' '.join(train_split[i])
	D_train.append(str1)

D_test = []
for i in range(len(test_split)):
	str1 = ' '.join(test_split[i])
	D_test.append(str1)

#%%
train['if positive'] = train[train.columns[0]].apply(lambda x: 1 if x >= 3 else 0)
test['if positive'] = test[test.columns[0]].apply(lambda x: 1 if x >= 3 else 0)
# print(train.head(10))
y_train = list(train[train.columns[3]])
y_test = list(test[test.columns[3]])

#%%
print(y_train)
#%%
#======================= TF-IDF Feature Selection =================================
# from sklearn.feature_extraction.text import TfidfVectorizer 
# from nltk.corpus import stopwords 
# tfidfconverter = TfidfVectorizer(max_features = 150, min_df=5, max_df=0.6, stop_words=stopwords.words('english')) 
# X_train = tfidfconverter.fit_transform(D_train).toarray()
# features = tfidfconverter.get_feature_names()
# print(tfidfconverter.get_feature_names())
# temp = TfidfVectorizer(vocabulary=features)
# X_test = temp.fit_transform(D_test).toarray()
# print(temp.get_feature_names())

#%%
#======================= CountVector Feature Selection =================================
countvectorizer = CountVectorizer(max_features = 150, min_df=5, max_df=0.6, stop_words=stopwords.words('english'))
X_train = countvectorizer.fit_transform(D_train).toarray()
features = countvectorizer.get_feature_names()
print(countvectorizer.get_feature_names())
temp = CountVectorizer(vocabulary=features)
X_test = temp.fit_transform(D_test).toarray()
print(temp.get_feature_names())


#%%
#======================= model training ====================================

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train[:train_size])  

#%%
y_pred = classifier.predict(X_test) 

#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))


#useless
#%%
# def AvgSen(sen_list,wv_model):
#     word_set = set(wv_model.wv.index2word)
#     X_avg = np.zeros([len(sen_list),50])
#     c=0
#     for sen in sen_list:
#         temp = np.zeros([50,])
#         nw=0
#         for w in sen:
#             if w in word_set:
#                 nw=nw+1
#                 temp = temp + wv_model[w]
#         X_avg[c] = temp/nw
#         c=c+1
#     return X_avg
#%%
# corpus = train[train.columns[2]].tolist()
# tokenized_sentences = [sentence.split() for sentence in corpus]
# model = word2vec.Word2Vec(tokenized_sentences, min_count=1)
# X_avg = AvgSen(sen_list,model)
# X_avg