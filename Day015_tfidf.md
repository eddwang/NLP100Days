## 作業目標：搭建一個TFIDF 模型

---

#### Reference:https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76


```python
import nltk
import numpy as np
documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'
```

## 首先我們做tokenize，並取出所有文件中的單詞


```python

tokenize_A = nltk.word_tokenize(documentA)
tokenize_B = nltk.word_tokenize(documentB)

uniqueWords = set(tokenize_A).union(set(tokenize_B)) ##所有文件中的單詞
```

## 計算每個文件中，所有uniqueWords出現的次數


```python
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in tokenize_A:
    numOfWordsA[word] += 1
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in tokenize_B:
    numOfWordsB[word] += 1
```


```python
numOfWordsA
```




    {'children': 0,
     'sat': 0,
     'went': 1,
     'walk': 1,
     'for': 1,
     'fire': 0,
     'around': 0,
     'the': 1,
     'a': 1,
     'out': 1,
     'man': 1}




```python
numOfWordsB
```




    {'children': 1,
     'sat': 1,
     'went': 0,
     'walk': 0,
     'for': 0,
     'fire': 1,
     'around': 1,
     'the': 2,
     'a': 0,
     'out': 0,
     'man': 0}



## 定義function:計算TF


```python
def computeTF(wordDict, tokenize_item):
    """
    wordDict : 文件內單詞對應出現數量的字典
    tokenize_item : 文件tokenize後的輸出
    """
    tfDict = {}
    bagOfWordsCount = len(tokenize_item) ## tokenize_item單詞數量
    for word, count in wordDict.items():
        tfDict[word] = count/bagOfWordsCount ##單詞在該文件出現的次數/該文件擁有的所有單詞數量
    return tfDict
```

## 定義function:計算IDF


```python
def computeIDF(documentsDict):
    """
    documentsDict:為一個list，包含所有文件的wordDict
    """
    import math
    N = len(documentsDict)
    
    idfDict = dict.fromkeys(documentsDict[0].keys(), 0)
    for document in documentsDict:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1 ## 計算單詞在多少文件中出現過
    
    for word, val in idfDict.items():
        idfDict[word] = np.log(N/val) ## 計算IDF，Log (所有文件的數目/包含這個單詞的文件數目)
    return idfDict
```

## 定義function:計算TFIDF


```python

def computeTFIDF(tf_item, idfs):
    tfidf = {}
    for word, val in tf_item.items():
        tfidf[word] = val * idfs[word]
    return tfidf


```


```python
tfA = computeTF(numOfWordsA, tokenize_A)
tfB = computeTF(numOfWordsB, tokenize_B)

idfs = computeIDF([numOfWordsA, numOfWordsB])


tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
```


```python
tfidfA
```




    {'children': 0.0,
     'sat': 0.0,
     'went': 0.09902102579427789,
     'walk': 0.09902102579427789,
     'for': 0.09902102579427789,
     'fire': 0.0,
     'around': 0.0,
     'the': 0.0,
     'a': 0.09902102579427789,
     'out': 0.09902102579427789,
     'man': 0.09902102579427789}




```python
tfidfB
```




    {'children': 0.11552453009332421,
     'sat': 0.11552453009332421,
     'went': 0.0,
     'walk': 0.0,
     'for': 0.0,
     'fire': 0.11552453009332421,
     'around': 0.11552453009332421,
     'the': 0.0,
     'a': 0.0,
     'out': 0.0,
     'man': 0.0}




```python

```
