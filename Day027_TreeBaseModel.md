### 作業目的: 實作樹型模型

在本次課程中實作了以Entropy計算訊息增益的決策樹模型，而計算訊息增益的方法除了Entropy只外還有Gini。因此本次作業希望讀者實作以Gini計算

訊息增益，且基於課程的決策樹模型建構隨機森林模型。

在作業資料夾中的`decision_tree_functions.py`檔案有在作業中實作的所有函式，在實作作業中可以充分利用已經寫好的函式

### Q1: 使用Gini計算訊息增益

$$
Gini = \sum_{i=1}^cp(i)(1-p(i)) = 1 - \sum_{i=1}^cp(i)^2
$$


```python
import pandas as pd
import numpy as np
import random
from decision_tree_functions import decision_tree, train_test_split
```


```python
# 使用與課程中相同的假資料

training_data = [
    ['Green', 3.1, 'Apple'],
    ['Red', 3.2, 'Apple'],
    ['Red', 1.2, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3.3, 'Lemon'],
    ['Yellow', 3.1, 'Lemon'],
    ['Green', 3, 'Apple'],
    ['Red', 1.1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
    ['Red', 1.2, 'Grape'],
]

header = ["color", "diameter", "label"]

df = pd.DataFrame(data=training_data, columns=header)
df.head()
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
      <th>color</th>
      <th>diameter</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Green</td>
      <td>3.1</td>
      <td>Apple</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Red</td>
      <td>3.2</td>
      <td>Apple</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Red</td>
      <td>1.2</td>
      <td>Grape</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Red</td>
      <td>1.0</td>
      <td>Grape</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Yellow</td>
      <td>3.3</td>
      <td>Lemon</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Gini impurity
def calculate_gini(data):
    
    #取的資料的label訊息
    ###<your code>###
    label_column = data[:, -1]
    
    #取得所有輸入資料的獨立類別與其個數
    ###<your code>###
    _, counts = np.unique(label_column, return_counts=True)
    
    #計算機率
    ###<your code>###
    probabilities = counts / counts.sum()
    
    #計算gini impurity
    ###<your code>###
    gini = sum(1-(probabilities**2))
    
    return gini
```


```python

```


```python
#分割資料集
###<your code>###
train_df, test_df=train_test_split(df, 0.2)
#以Gini inpurity作為metric_function訓練決策樹
tree = decision_tree(metric_function=calculate_gini)
tree.fit(train_df)
```




    {'diameter <= 1.2': ['Grape', {'color = Yellow': ['Lemon', 'Apple']}]}




```python
# 以建構好的樹進行預測
sample = test_df.iloc[0]
tree.pred(sample,tree.sub_tree)
###<your code>###
```




    'Apple'




```python
sample
```




    color       Green
    diameter      3.1
    label       Apple
    Name: 0, dtype: object



### Q2: 實作隨機森林
利用決策樹來實作隨機森林模型，讀者可參考隨機森林課程講義。

此份作業只要求讀者實作隨機sample訓練資料，而隨機sample特徵進行訓練的部分，讀者可以參考`decision_tree_functions.py`中的`get_potential_splits`與`decision_tree`部分(新增參數`random_features`)


```python
class random_forest():
    '''Random forest model
    Parameters
    ----------
    n_boostrap: int
        number of samples to sample to train indivisual decision tree
    n_tree: int
        number of trees to form a forest
    '''
    
    def __init__(self, n_bootstrap, n_trees, task_type, min_samples, max_depth, metric_function, n_features=None):
        self.n_bootstrap = n_bootstrap
        self.n_trees = n_trees
        self.task_type = task_type
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.metric_function = metric_function
        self.n_features = n_features
    
    def bootstrapping(self, train_df,n_bootstrap ):
        indices = random.sample(train_df.index.tolist(),n_bootstrap)
        df_bootstrapped = train_df.loc[indices]
        return df_bootstrapped
    
    def fit(self, train_df):
        self.forest = []
        for i in range(self.n_trees):
            df_bootstrapped=self.bootstrapping(train_df,self.n_bootstrap)
            tree = decision_tree(metric_function=self.metric_function)
            tree.fit(df_bootstrapped)
            self.forest.append(tree)
        
        ###<your code>###           
        return self.forest
    
    def pred(self, test_df):
        df_predictions = {}
        
        ###<your code>###
        for i in range(len(self.forest)):
            sub_tree = self.forest[i].sub_tree
            df_predictions[i] = [self.forest[i].pred(test_df.iloc[j],sub_tree) for j in range(test_df.shape[0])]
        df_predictions = pd.DataFrame(df_predictions)
        
        
        
        #majority voting
        ###<your code>###
        random_forest_predictions= df_predictions.apply(lambda x : x.value_counts().sort_values(ascending=False).index.tolist()[0],axis=1)
        return random_forest_predictions
```


```python
train_df.shape
```




    (8, 3)




```python
train_df, test_df = train_test_split(df, 0.2)

#建立隨機森林模型
###<your code>###
forest = random_forest(4, n_trees=7,task_type='classification', min_samples=1, max_depth=10, metric_function=calculate_gini)
forest.fit(train_df)
```




    [<decision_tree_functions.decision_tree at 0x7f29b8ec8fd0>,
     <decision_tree_functions.decision_tree at 0x7f29b8319a50>,
     <decision_tree_functions.decision_tree at 0x7f29b83539d0>,
     <decision_tree_functions.decision_tree at 0x7f29b82ff7d0>,
     <decision_tree_functions.decision_tree at 0x7f29b82ff5d0>,
     <decision_tree_functions.decision_tree at 0x7f29e92c8750>,
     <decision_tree_functions.decision_tree at 0x7f29b82ff910>]




```python
forest.pred(test_df)
```




    0    Grape
    1    Apple
    dtype: object




```python
test_df
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
      <th>color</th>
      <th>diameter</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Red</td>
      <td>1.1</td>
      <td>Grape</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Green</td>
      <td>3.1</td>
      <td>Apple</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
