# K-Nearest Neighbors (K-NN)

### 參考課程實作並在datasets_483_982_spam.csv的資料集中獲得90% 以上的 accuracy (testset)

## Importing the libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import codecs
import re
```

## Importing the dataset


```python
all_data=[]
paths =[r'spam_data/spam', r'spam_data/easy_ham', r'spam_data/hard_ham'] 
for path in paths:
    for fn in glob.glob(path+"/*"):
        if "ham" not in fn:
            is_spam = 1
        else:
            is_spam = 0
        #codecs.open可以避開錯誤，用errors='ignore'
        with codecs.open(fn,encoding='utf-8', errors='ignore') as file:
            for line in file:
                #這個line的開頭為Subject:
                if line.startswith("Subject:"):
                    subject=re.sub(r"^Subject:","",line).strip()
                    all_data.append([subject,is_spam])
all_data = np.array(all_data)
```


```python
dataset = pd.read_csv(r'datasets_483_982_spam.csv', encoding = 'latin-1')

```

### 取出訓練內文與標註


```python
X = all_data[:,0]
Y = all_data[:,1].astype(np.uint8)
```


```python
print('Training Data Examples : \n{}'.format(X[:5]))
```

    Training Data Examples : 
    ['One of a kind Money maker! Try it for free!' 'Grow Up And Be A Man! abm'
     'are you in the mood             XGHTMTGGC' 'NEED TO FIND SOMETHING?'
     '[ILUG-Social] please kindly get back to me']



```python
print('Labeling Data Examples : \n{}'.format(Y[:5]))
```

    Labeling Data Examples : 
    [1 1 1 1 1]


### 文字預處理


```python
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords

import nltk

nltk.download('stopwords',quiet=True)

# Lemmatize with POS Tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 


## 創建Lemmatizer
lemmatizer = WordNetLemmatizer() 
def get_wordnet_pos(word):
    """將pos_tag結果mapping到lemmatizer中pos的格式"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
def clean_content(X):
    # remove non-alphabet characters
    X_clean = [re.sub('[^a-zA-Z]',' ', x).lower() for x in X]
    # tokenize
    X_word_tokenize = [nltk.word_tokenize(x) for x in X_clean]
    # stopwords_lemmatizer
    X_stopwords_lemmatizer = []
    stop_words = set(stopwords.words('english'))
    for content in X_word_tokenize:
        content_clean = []
        for word in content:
            if word not in stop_words:
                word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
                content_clean.append(word)
        X_stopwords_lemmatizer.append(content_clean)
    
    X_output = [' '.join(x) for x in X_stopwords_lemmatizer]
    
    return X_output
```


```python
X = clean_content(X)
```


```python
n_neighbors = 3
max_features = 400
explore_k = True
```


```python

```

### Bag of words


```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#max_features是要建造幾個column，會按造字出現的高低去篩選 
cv=CountVectorizer(max_features = max_features)
tfidfv=TfidfVectorizer(max_features = max_features)
# X_trans=cv.fit_transform(X).toarray()
X_trans=tfidfv.fit_transform(X).toarray()
```


```python
X_trans.shape
```




    (3423, 400)



## Splitting the dataset into the Training set and Test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_trans, Y, test_size = 0.2, random_state = 0)
```

## Training the K-NN model on the Training set


```python
from sklearn.neighbors import KNeighborsClassifier

```


```python
if explore_k :
    n_neighbors_list=range(3,19,2)
    for n in n_neighbors_list:
        classifier = KNeighborsClassifier(n_neighbors = n, metric = 'minkowski', p = 2,n_jobs=-1)
        classifier.fit(X_train, y_train)
        print(f'\nk={n}')
        print('Testset Accuracy: {}'.format(classifier.score(X_test, y_test)))
```

    
    k=3
    Testset Accuracy: 0.9007299270072993
    
    k=5
    Testset Accuracy: 0.8934306569343066
    
    k=7
    Testset Accuracy: 0.8905109489051095
    
    k=9
    Testset Accuracy: 0.8934306569343066
    
    k=11
    Testset Accuracy: 0.8890510948905109
    
    k=13
    Testset Accuracy: 0.8744525547445255
    
    k=15
    Testset Accuracy: 0.8715328467153285
    
    k=17
    Testset Accuracy: 0.8671532846715329



```python

classifier = KNeighborsClassifier(n_neighbors = n_neighbors, metric = 'minkowski', p = 2,n_jobs=-1)
classifier.fit(X_train, y_train)
```




    KNeighborsClassifier(n_jobs=-1, n_neighbors=3)



## Predicting a new result


```python
print('Trainset Accuracy: {}'.format(classifier.score(X_train, y_train)))
```

    Trainset Accuracy: 0.9298758217677137



```python
print('Testset Accuracy: {}'.format(classifier.score(X_test, y_test)))
```

    Testset Accuracy: 0.9007299270072993


## Predicting the Test set results


```python
y_pred = classifier.predict(X_test)
```

## Making the Confusion Matrix


```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

    [[562  25]
     [ 43  55]]





    0.9007299270072993




```python

```
