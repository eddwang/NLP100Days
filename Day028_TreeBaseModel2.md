### 作業目的: 使用樹型模型進行文章分類

本次作業主利用[Amazon Review data中的All Beauty](https://nijianmo.github.io/amazon/index.html)來進行review評價分類(文章分類)

資料中將review分為1,2,3,4,5分，而在這份作業，我們將評論改分為差評價、普通評價、優良評價(1,2-->1差評、3-->2普通評價、4,5-->3優良評價)

### 載入套件


```python
import json
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import string
```


```python
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Beauty.json.gz
!gunzip All_Beauty.json.gz
```

    --2020-12-16 19:47:07--  http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Beauty.json.gz
    Resolving deepyeti.ucsd.edu (deepyeti.ucsd.edu)... 169.228.63.50
    Connecting to deepyeti.ucsd.edu (deepyeti.ucsd.edu)|169.228.63.50|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 47350910 (45M) [application/octet-stream]
    Saving to: ‘All_Beauty.json.gz’
    
    All_Beauty.json.gz  100%[===================>]  45.16M  1.70MB/s    in 27s     
    
    2020-12-16 19:47:35 (1.67 MB/s) - ‘All_Beauty.json.gz’ saved [47350910/47350910]
    
    gzip: All_Beauty.json already exists; do you wish to overwrite (y or n)? ^C


### 資料前處理
文本資料較為龐大，這裡我們取前10000筆資料來進行作業練習


```python
#load json data
all_reviews = []
###<your code>###
with open("./All_Beauty.json", "r") as f:
    for review in f:
        all_reviews.append(json.loads(review))
all_reviews[0]
```




    {'overall': 1.0,
     'verified': True,
     'reviewTime': '02 19, 2015',
     'reviewerID': 'A1V6B6TNIC10QE',
     'asin': '0143026860',
     'reviewerName': 'theodore j bigham',
     'reviewText': 'great',
     'summary': 'One Star',
     'unixReviewTime': 1424304000}




```python
#parse label(overall) and corpus(reviewText)
corpus = []
labels = []

###<your code>###
for item in all_reviews[:10000]:
    text = item['reviewText'] if 'reviewText' in item.keys() else ""
    if text == "" : continue
    corpus.append(text)
    label = item['overall']
    if label in [1,2]:
        label = 1
    elif label in [3]:
        label = 2
    elif label in [4,5]:
        label = 3
    else:
        label = -1
    labels.append(label)
    
#transform labels: 1,2 --> 1 and 3 --> 2 and 4,5 --> 3

###<your code>###
```


```python
#preprocessing data
#remove email address, punctuations, and change line symbol(\n)
for i,txt in enumerate(corpus):
    txt = re.sub("["+string.punctuation+"]","",txt)
    txt = re.sub(r"\n"," ",txt)
    corpus[i] = txt
###<your code>###
```


```python
#split corpus and label into train and test
###<your code>###
x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)
len(x_train), len(x_test), len(y_train), len(y_test)
```




    (7996, 1999, 7996, 1999)




```python

```


```python
#change corpus into vector
#you can use tfidf or BoW here

###<your code>###
vectorizer = TfidfVectorizer()
vectorizer.fit(x_train)
#transform training and testing corpus into vector form
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)
```

### 訓練與預測


```python
#build classification model (decision tree, random forest, or adaboost)
#start training

###<your code>###
tree = DecisionTreeClassifier(max_depth=6)
tree.fit(x_train, y_train)
```




    DecisionTreeClassifier(max_depth=6)




```python
#start inference
y_pred = tree.predict(x_test)
```


```python
#calculate accuracy
print(f"Accuracy: {tree.score(x_test, y_test)}")
```

    Accuracy: 0.9074537268634317



```python
#calculate confusion matrix, precision, recall, and f1-score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               1       0.63      0.19      0.30       124
               2       0.00      0.00      0.00        73
               3       0.91      0.99      0.95      1802
    
        accuracy                           0.91      1999
       macro avg       0.52      0.40      0.42      1999
    weighted avg       0.86      0.91      0.88      1999
    
    [[  24    0  100]
     [   4    0   69]
     [  10    2 1790]]


由上述資訊可以發現, 模型在好評的準確度高(precision, recall都高), 而在差評的部分表現較不理想, 在普通評價的部分大部分跟差評搞混,
同學可以試著學習到的各種方法來提升模型的表現


```python

```
