
#%%
import pandas as pd
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
train_size = 80000
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# news data 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%%
r1 = '[0-9’!１２３４５６７８９０"#$%&\'()（）*+,-/:;<=>?@，。?★、…【】《》＊;「」？“”‘’！：[\\]^_`{|}~‧]+'
train[train.columns[1]] = train[train.columns[1]].apply(lambda x: re.sub(r1,'',x))

#%%
# ============== pos tagging ==============
train_split = []
test_split = []

for i in range(train_size):
    row = train.loc[i]
    temp = nltk.pos_tag(word_tokenize(row[2]))
    nounVerb = []
    required_type = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']
    for i in temp:
        if i[1] in required_type and i[0] != 'Reuters':
            nounVerb.append(lemmatizer.lemmatize(i[0]))
    train_split.append(nounVerb)

# test part
for i in range(7599):
    row = test.loc[i]
    temp = nltk.pos_tag(word_tokenize(row[2]))
    nounVerb = []
    required_type = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']
    for i in temp:
        if i[1] in required_type and i[0] != 'Reuters':
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
y_train = list(train[train.columns[0]])
y_test = list(test[test.columns[0]])

#%%
#======================= Feature Selection =================================
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.corpus import stopwords 
tfidfconverter = TfidfVectorizer(max_features = 1500, min_df=5, max_df=0.6, stop_words=stopwords.words('english')) 
X_train = tfidfconverter.fit_transform(D_train).toarray()
features = tfidfconverter.get_feature_names()
print(tfidfconverter.get_feature_names())
temp = TfidfVectorizer(vocabulary=features)
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
print(len(y_pred))
#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))
