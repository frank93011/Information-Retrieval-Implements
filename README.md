# Information Retrieval Implements

[TOC]


Multiple information retrieval and data mining techinique implements via python.
If you want detailed report of each project please see also the pdf in folders.

## Amazon Review Sentiment Analysis
### Requirement
* python 3.6
* Pandas
* nltk
* sklearn(RandomForest)
### Dataset
Amazon Review Full Score Dataset (constructed by Xiang Zhang)

The dataset train.csv and test.csv contain all the training
samples as comma-sparated values. There are 3 columns
in them, corresponding to class index (1 to 5), review title
and review text. 

> ex: ( 4, Surprisingly delightful, This is a fast read filled with
unexpected humour and profound insights into the art of politics and
policy. In brief, it is sly, wry, and wise. )

## Aspect Level Sentimental Analysis 
### Requirement
* python 3.6
* Pandas
* nltk
* sklearn(RandomForest)
* spaCy
### Dataset
Review of Restaurant

contains the release of a sample of English Foursquare restaurant reviews annotated with Aspect Based Sentiment annotations,
together with an evaluation script enabling assessment of performances of Aspect Based Sentiment detection systems.
These Foursquare reviews have been annotated using the SemEval2016 ABSA challenge annotation schema.
Link to  SemEval2016 challenge web page: http://alt.qcri.org/semeval2016/task5/
The annotation guidelines are available here: http://alt.qcri.org/semeval2016/task5/data/uploads/absa2016_annotationguidelines.pdf


> ex: They tried to change us more than double service charge on our bill.


## News Topic Prediction
### Requirement
* python 3.6
* Pandas
* nltk
* sklearn(RandomForest)
### Dataset
AG collection of news articles (constructed by Xiang Zhang)

The AG's news topic classification dataset is constructed by
choosing 4 largest classes from the original corpus. Each
class contains 30,000 training samples and 1,900 testing
samples. The total number of training samples is 120,000
and testing 7,600. The classes are: World, Sports, Business
and Sci/Tech.

> ex: ( 4, Surprisingly delightful, This is a fast read filled with
unexpected humour and profound insights into the art of politics and
policy. In brief, it is sly, wry, and wise. )

## Porter Stemming Implement
### Requirement
* python 3.6
* nltk
* re
### Dataset
Any English news document  

### How to Use
程式會讀入在同一資料夾內檔名為”input.txt”的文字檔，並產生出”output.txt”。
Output.txt 中會列出所有 stemming 過後且不包含 stopwords 的單字們，並用逗號隔開

## tf-idf Implement
### Requirement
* python 3.6
* nltk
* re
* numpy
* collections
* pattern
### Dataset
English news document collections

### How to Use
程式會讀入在同一資料夾內路徑為”IRTM”的資料夾，並開啟裡面所有的檔案作為 df 計算的
依據，程式最終會產出符合 json 格式的資料: “dictionary.txt”, “Doc1.txt”, “Doc2.txt”
於執行檔同一個目錄中。
再執行時最後會印出 Doc1.txt 與 Doc2.txt 的 Cosine Similarity

## Chisquare Selection Implement
### Requirement
* python 3.6
* nltk
* re
* numpy
* collections
* pattern
### Dataset
English news document collections

### How to Use
程式會讀入 training.txt 與裡面包含的 training data，在同一資料夾內路徑為”IRTM”的資
料夾，並開啟裡面所有的檔案作為計算的依據，程式最終會產出 csv 檔案，包含 id 與 value 兩
項

## Centroid Clustering Implement
### Requirement
* python 3.6
* nltk
* re
* numpy
* collections
* pattern
* heapq
### Dataset
English news document collections

### How to Use
程式會讀入在同一資料夾內路徑為”IRTM”的資料夾，並開啟裡面所有的檔案作為計算的依
據，程式最終會產出選定的 txt 檔案
