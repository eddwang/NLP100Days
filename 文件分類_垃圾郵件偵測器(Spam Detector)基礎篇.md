```python
#題目: 將已整理好的文件以機器學習方式分辨是否為垃圾郵件
#說明：輸入文件已處理過，為一D乘V(V=48)+1矩陣，D代表電郵數，V代表選出來(判斷是否垃圾)的字(特徵)，
#所以我們是用48個特徵來判斷。列中每行表達的特徵值(feature)=出現次數 / 該電郵總字數 * 100，
#最後一行是標註(Label)是否為垃圾郵件。請用ML方法開發出垃圾郵件偵測器並算出預測準確度
#延伸:可用不同ML分類法，可準備自己的垃圾郵件做預處理。
#範例程式檔名: spam_nb_垃圾郵件偵測器.py，以Naïve Bayes方式完成
#模組: sklearn, pandas, numpy
#輸入檔：spambase.data
#成績：辨識百分率
```


```python
from __future__ import print_function, division
from builtins import range

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
np.random.seed(111)
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
```


```python
# 註: 理論上 multinomial NB 是針對出現次數 "counts", 但文件上說對出現比率 "word proportions"也適合

data = pd.read_csv('spambase.data').values # use pandas for convenience
np.random.shuffle(data) # shuffle each row in-place, but preserve the row

X = data[:,:48]
Y = data[:,-1]
```


```python
# 最後80列用作測試
Xtrain = X[:-80,]
Ytrain = Y[:-80,]
Xtest = X[-80:,]
Ytest = Y[-80:,]
```


```python
# Decision Tree 的準確度
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for Decision Tree:", model.score(Xtest, Ytest))
```

    Classification rate for Decision Tree: 0.875



```python
# AdaBoost 的準確度
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))
```

    Classification rate for AdaBoost: 0.8875



```python

```


```python
from sklearn.decomposition import PCA
```


```python
from sklearn.tree import DecisionTreeClassifier

for n in range(2,40):
    pca = PCA(n_components=n,random_state=38)
    pca.fit(Xtrain)
    pcaXtrain=pca.transform(Xtrain)
    pcaXtest=pca.transform(Xtest)

    model = DecisionTreeClassifier()
    model.fit(pcaXtrain, Ytrain)
    print("n={0}\tClassification rate for Decision Tree:{1}".format(n,model.score(pcaXtest, Ytest)))
```

    n=2	Classification rate for Decision Tree:0.7375
    n=3	Classification rate for Decision Tree:0.7625
    n=4	Classification rate for Decision Tree:0.85
    n=5	Classification rate for Decision Tree:0.85
    n=6	Classification rate for Decision Tree:0.8375
    n=7	Classification rate for Decision Tree:0.8625
    n=8	Classification rate for Decision Tree:0.8625
    n=9	Classification rate for Decision Tree:0.8375
    n=10	Classification rate for Decision Tree:0.8125
    n=11	Classification rate for Decision Tree:0.8875
    n=12	Classification rate for Decision Tree:0.875
    n=13	Classification rate for Decision Tree:0.9
    n=14	Classification rate for Decision Tree:0.8875
    n=15	Classification rate for Decision Tree:0.9
    n=16	Classification rate for Decision Tree:0.875
    n=17	Classification rate for Decision Tree:0.8875
    n=18	Classification rate for Decision Tree:0.8875
    n=19	Classification rate for Decision Tree:0.8875
    n=20	Classification rate for Decision Tree:0.875
    n=21	Classification rate for Decision Tree:0.9
    n=22	Classification rate for Decision Tree:0.8625
    n=23	Classification rate for Decision Tree:0.9
    n=24	Classification rate for Decision Tree:0.8625
    n=25	Classification rate for Decision Tree:0.85
    n=26	Classification rate for Decision Tree:0.825
    n=27	Classification rate for Decision Tree:0.85
    n=28	Classification rate for Decision Tree:0.8125
    n=29	Classification rate for Decision Tree:0.85
    n=30	Classification rate for Decision Tree:0.8375
    n=31	Classification rate for Decision Tree:0.85
    n=32	Classification rate for Decision Tree:0.8625
    n=33	Classification rate for Decision Tree:0.8625
    n=34	Classification rate for Decision Tree:0.875
    n=35	Classification rate for Decision Tree:0.8625
    n=36	Classification rate for Decision Tree:0.8625
    n=37	Classification rate for Decision Tree:0.875
    n=38	Classification rate for Decision Tree:0.8875
    n=39	Classification rate for Decision Tree:0.8875



```python
from sklearn.tree import DecisionTreeClassifier

for n in range(2,40):
    pca = PCA(n_components=n,random_state=38)
    pca.fit(Xtrain)
    pcaXtrain=pca.transform(Xtrain)
    pcaXtest=pca.transform(Xtest)

    model = AdaBoostClassifier()
    model.fit(pcaXtrain, Ytrain)
    print("n={0}\tClassification rate for Decision Tree:{1}".format(n,model.score(pcaXtest, Ytest)))
```

    n=2	Classification rate for Decision Tree:0.8
    n=3	Classification rate for Decision Tree:0.7875
    n=4	Classification rate for Decision Tree:0.8375
    n=5	Classification rate for Decision Tree:0.875
    n=6	Classification rate for Decision Tree:0.875
    n=7	Classification rate for Decision Tree:0.8875
    n=8	Classification rate for Decision Tree:0.875
    n=9	Classification rate for Decision Tree:0.8625
    n=10	Classification rate for Decision Tree:0.9125
    n=11	Classification rate for Decision Tree:0.925
    n=12	Classification rate for Decision Tree:0.9
    n=13	Classification rate for Decision Tree:0.925
    n=14	Classification rate for Decision Tree:0.9375
    n=15	Classification rate for Decision Tree:0.9
    n=16	Classification rate for Decision Tree:0.9375
    n=17	Classification rate for Decision Tree:0.9125
    n=18	Classification rate for Decision Tree:0.925
    n=19	Classification rate for Decision Tree:0.9125
    n=20	Classification rate for Decision Tree:0.9125
    n=21	Classification rate for Decision Tree:0.9
    n=22	Classification rate for Decision Tree:0.9125
    n=23	Classification rate for Decision Tree:0.9125
    n=24	Classification rate for Decision Tree:0.9
    n=25	Classification rate for Decision Tree:0.9125
    n=26	Classification rate for Decision Tree:0.9
    n=27	Classification rate for Decision Tree:0.9
    n=28	Classification rate for Decision Tree:0.9125
    n=29	Classification rate for Decision Tree:0.9125
    n=30	Classification rate for Decision Tree:0.8875
    n=31	Classification rate for Decision Tree:0.9125
    n=32	Classification rate for Decision Tree:0.9
    n=33	Classification rate for Decision Tree:0.9125
    n=34	Classification rate for Decision Tree:0.9125
    n=35	Classification rate for Decision Tree:0.9125
    n=36	Classification rate for Decision Tree:0.9125
    n=37	Classification rate for Decision Tree:0.9
    n=38	Classification rate for Decision Tree:0.925
    n=39	Classification rate for Decision Tree:0.925


整體來說，AdaBoost 的表現較穩定，降維可以有更好的表現

最後，將垃圾郵件偵測器實做出來


```python

def Spam_Detector(train_data,pred_data):
    X = train_data[:,:48]
    Y = train_data[:,-1]
    pca = PCA(n_components=16,random_state=38)
    pca.fit(X)
    predX=pca.transform(pred_data)
    model = AdaBoostClassifier()
    
    model.fit(pca.transform(X), Y)
    return model.predict(predX)

Spam_Detector(data[:-20,:],data[-20:,:48])   
```




    array([1., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0.,
           1., 0., 0.])




```python
data[-20:,-1]
```




    array([1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1.,
           0., 0., 0.])




```python

```
