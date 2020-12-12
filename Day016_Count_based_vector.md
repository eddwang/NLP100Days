## 作業目標: 透過思考與回答以更加了解計數方法的詞向量

### 請問詞庫手法會有什麼樣的優缺點？

詞庫手法為創建包含大量字詞的詞庫，將相同意思字詞(同義字)或相似意思字詞(相似字)分類在相同群組。

優點:
比較精準，誤解的機率低

缺點:
不存在詞庫的字無法判別，同時詞庫的意義也會隨著時間漸漸失效

### 請問共現矩陣有什麼樣的優缺點？ 

根據分佈假說，相似的字詞會有類似的上下文，因此我們可以透過計數周圍(window)的字詞來表達特定字詞的向量。

優點:
可以得知上下文代表的意義，不限於一個字

缺點:
所需的資料量大，同時也會受到高頻詞彙的影響導致誤判

### 請問為何需要對共現矩陣或PPMI進行SVD降維?

避免因為空值太多造成稀疏矩陣帶來的準確度下降

### 實作cosine similarity

在比較兩個詞向量的相似度時可以使用cosine similarity:
$$
similarity(x,y) = \frac{x \cdot y}{||x||||y||} = \frac{x_1y_1+...+x_ny_n}{\sqrt{x_1^2+...+x_n^2}\sqrt{y_1^2+...+y_n^2}}
$$

請實作cosine similarity並計算共現矩陣課程範例中you向量([0,1,0,0,0,0,0])與I([0,1,0,1,0,0,0])向量的相似度


```python
import numpy as np
I = np.array([0,1,0,0,0,0,0])
You = np.array([0,1,0,1,0,0,0])

def cos_similarity(x, y, eps=1e-8):
    ### your code ###
    return np.dot(x,y) /(np.linalg.norm(x) * np.linalg.norm(y))

print(f'Similarity: {cos_similarity(I, You)}')
```

    Similarity: 0.7071067811865475



```python

```
