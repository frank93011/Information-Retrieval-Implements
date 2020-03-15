#%%
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re

#%%
input = open("input.txt","r")   #read txt file
txt = input.read().lower()
txt = re.sub(r'[^\w\s]','',txt) #clear all non-word
#%%
word = txt.split()
new = ""
stop = set(stopwords.words('english'))
ps = PorterStemmer()    #using porter stemming algorithm
for w in word:
    if w not in stop:
        if(w != word[-1]):
            new += ps.stem(w)+","
        else:
            new += ps.stem(w)

#%%
output = open("output.txt", 'w')
output.write(new)
output.close()
#%%
