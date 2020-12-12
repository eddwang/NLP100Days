## 本日課程-文字預處理，部分內容前面章節可能提過，這裡會將前處理所需技巧串起


```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
#tsv是指用tab分開字元的檔案
dataset=pd.read_csv('movie_feedback.csv', header=None, encoding='Big5')
X = dataset[0].values
Y = dataset[1].values
```


```python
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>the rock is destined to be the 21st century's ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>the gorgeously elaborate continuation of " the...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>effective but too-tepid biopic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>if you sometimes like to go to the movies to h...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>emerges as something rare , an issue movie tha...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10657</th>
      <td>a terrible movie that some people will neverth...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10658</th>
      <td>there are many definitions of 'time waster' bu...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10659</th>
      <td>as it stands , crocodile hunter has the hurrie...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10660</th>
      <td>the thing looks like a made-for-home-video qui...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10661</th>
      <td>enigma is well-made , but it's just too dry an...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10662 rows × 2 columns</p>
</div>



---


```python
print('review before preprocessing : {}'.format(X[0]))
```

    review before preprocessing : the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal . 


## 運用re.sub去除部分字元


```python
import re 
# 去除a-zA-Z以外的字元，並將他們取代為空格' '
review=re.sub('[^a-z^A-Z ]',' ',X[0])
```


```python
print('review after re.sub : {}'.format(review))
```

    review after re.sub : the rock is destined to be the   st century s new   conan   and that he s going to make a splash even greater than arnold schwarzenegger   jean claud van damme or steven segal   


## 將所有字母轉為小寫:因為大部分情境區分大小寫並不能提供而外訊息，如CV內顏色無法提供額外訊息時我們會將圖像轉為灰階，藉此降低複雜度


```python
#把全部變成小寫
review=review.lower()
print('review after lower : {}'.format(review))
```

    review after lower : the rock is destined to be the   st century s new   conan   and that he s going to make a splash even greater than arnold schwarzenegger   jean claud van damme or steven segal   


## 斷詞


```python
import nltk
#把review裡面的單字切開
print('review after split : {}'.format(review.split()))
```

    review after split : ['the', 'rock', 'is', 'destined', 'to', 'be', 'the', 'st', 'century', 's', 'new', 'conan', 'and', 'that', 'he', 's', 'going', 'to', 'make', 'a', 'splash', 'even', 'greater', 'than', 'arnold', 'schwarzenegger', 'jean', 'claud', 'van', 'damme', 'or', 'steven', 'segal']


* tokenize 相較於split會是更好的選擇，如 split 無法分開 word. 這種case


```python
review = nltk.word_tokenize(review)
print('review after tokenize : {}'.format(review))
```

    review after tokenize : ['the', 'rock', 'is', 'destined', 'to', 'be', 'the', 'st', 'century', 's', 'new', 'conan', 'and', 'that', 'he', 's', 'going', 'to', 'make', 'a', 'splash', 'even', 'greater', 'than', 'arnold', 'schwarzenegger', 'jean', 'claud', 'van', 'damme', 'or', 'steven', 'segal']


## stopwords: 移除贅字，此步驟為前處理的重要步驟之一，過多的贅字不僅無法提供更多訊息，還會干擾到模型的訓練


```python
#處理文字，有建立好的文字褲會幫我們移除不想要的文字
import nltk
nltk.download('stopwords',quiet=True)
```




    True




```python
review=[word for word in review if not word in set(stopwords.words('english'))]
print('review after removeing stopwords : {}'.format(review))
```

    review after removeing stopwords : ['rock', 'destined', 'st', 'century', 'new', 'conan', 'going', 'make', 'splash', 'even', 'greater', 'arnold', 'schwarzenegger', 'jean', 'claud', 'van', 'damme', 'steven', 'segal']


## Stemming: 詞幹提取
 * ex. loves,loved都變成love
 * 中文沒有詞幹提取的需求


```python
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
review=[ps.stem(word) for word in review]
```


```python
print('review after stemming : {}'.format(review))
```

    review after stemming : ['rock', 'destin', 'st', 'centuri', 'new', 'conan', 'go', 'make', 'splash', 'even', 'greater', 'arnold', 'schwarzenegg', 'jean', 'claud', 'van', 'damm', 'steven', 'segal']


## 練習清理所有的句子


```python
#dataset=pd.read_csv('movie_feedback.csv',encoding = 'Big5',names=['feedback', 'label'] )
X = dataset[0].values
```


```python
corpus=[]
row=len(X)
for i in range(0,row):
    review=re.sub('[^a-zA-Z]',' ',X[i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    ## 這裡先不用stopwords 因為 review中很多反定詞會被移掉 如isn't good, 會變成 good
    review=[ps.stem(word) for word in review ]
    review=' '.join(review)
    corpus.append(review)
```

## 轉bag-of-words vector


```python
from sklearn.feature_extraction.text import CountVectorizer
#Creating bag of word model
#tokenization(符號化)
from sklearn.feature_extraction.text import CountVectorizer
#max_features是要建造幾個column，會按造字出現的高低去篩選 
cv = CountVectorizer(max_features=1500)
#toarray是建造matrixs
#X現在為sparsity就是很多零的matrix
X_ = cv.fit_transform(corpus).toarray()
Y_ = dataset[1].values
```

## 選擇練習: 將處理好數據放入 naive_bayes模型，並預測評論為正向或負面，詳細原理之後章節會解釋。

## Training


```python

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size = 0.1)

# Feature Scaling

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

```




    GaussianNB()



## Inference


```python

message='I really like this movie!!'
## 要使用一樣的前處理
review=re.sub('[^a-zA-Z]',' ',message)
review=review.lower()
review=review.split()
ps=PorterStemmer()
review=[ps.stem(word) for word in review]
review = ' '.join(review)
input_ = cv.transform([review]).toarray()
prediction = classifier.predict(input_)


```


```python
prediction ## 1代表正向評價
```




    array([0])




```python
message='A terrible movie  !!'
review=re.sub('[^a-zA-Z]',' ',message)
review=review.lower()
review=review.split()
ps=PorterStemmer()
review=[ps.stem(word) for word in review]
review = ' '.join(review)
input_ = cv.transform([review]).toarray()
prediction = classifier.predict(input_)


```


```python
prediction ## 0代表負面評價
```




    array([0])




```python

```
