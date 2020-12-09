### 作業目標: 了解N-Gram如何藉由文本計算機率

### 為何需要使用馬可夫假設來簡化語言模型的計算?

原本的語言模型利用貝氏定理計算機率時為:
$$
\begin{aligned}
&W = (W_1W_2W_3W_4…W_m) \\
&P(W_1,W_2,W_3,W_4,…,W_m) = P(W_1)*P(W_2|W_1)*P(W_3|W_1,W_2)*…*P(W_m|W_1,…,W_{m-1})
\end{aligned}
$$

為何需要引入馬可夫假設使機率簡化為:
$$
P(W_m|W_1,W_2,W_3,…,W_{m-1}) = P(W_m|W_{m-n+1},W_{m-n+2},…,W_{m-1})
$$


```python
'''
###<your answer>###

'''
'''
避免出現率太極端導致機率過小
'''

```




    '\n避免出現率太極端導致機率過小\n'



### 以Bigram模型下判斷語句是否合理

已知的機率值有
1. p(i|_start_) = 0.25
2. p(english|want) = 0.0011
3. p(food|english) = 0.5
4. p(_end_|food) = 0.68
5. P(want|_start_) = 0.25
6. P(english|i) = 0.0011


```python
import numpy as np
import pandas as pd
words = ['i', 'want', 'to', 'eat', 'chinese', 'food', 'lunch', 'spend']
word_cnts = np.array([2533, 927, 2417, 746, 158, 1093, 341, 278]).reshape(1, -1)
df_word_cnts = pd.DataFrame(word_cnts, columns=words)
df_word_cnts
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
      <th>i</th>
      <th>want</th>
      <th>to</th>
      <th>eat</th>
      <th>chinese</th>
      <th>food</th>
      <th>lunch</th>
      <th>spend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2533</td>
      <td>927</td>
      <td>2417</td>
      <td>746</td>
      <td>158</td>
      <td>1093</td>
      <td>341</td>
      <td>278</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 記錄當前字與前一個字詞的存在頻率
bigram_word_cnts = [[5, 827, 0, 9, 0, 0, 0, 2], [2, 0, 608, 1, 6, 6, 5, 1], [2, 0, 4, 686, 2, 0, 6, 211],
                    [0, 0, 2, 0, 16, 2, 42, 0], [1, 0, 0, 0, 0, 82, 1, 0], [15, 0, 15, 0, 1, 4, 0, 0],
                    [2, 0, 0, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]]

df_bigram_word_cnts = pd.DataFrame(bigram_word_cnts, columns=words, index=words)
df_bigram_word_cnts
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
      <th>i</th>
      <th>want</th>
      <th>to</th>
      <th>eat</th>
      <th>chinese</th>
      <th>food</th>
      <th>lunch</th>
      <th>spend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>i</th>
      <td>5</td>
      <td>827</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>want</th>
      <td>2</td>
      <td>0</td>
      <td>608</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>to</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>686</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>211</td>
    </tr>
    <tr>
      <th>eat</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>16</td>
      <td>2</td>
      <td>42</td>
      <td>0</td>
    </tr>
    <tr>
      <th>chinese</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>82</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>food</th>
      <td>15</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>lunch</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>spend</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



由上表可知當前一個字詞(列)是want的時候, 當前字詞(行)是to的頻率在文本中共有608次


```python
#請根據給出的總詞頻(df_word_cnts)與bigram模型的詞頻(df_bigram_word_cnts)計算出所有詞的配對機率(ex:p(want|i))
df_bigram_prob = df_bigram_word_cnts.copy()

###<your code>###
for w in df_bigram_prob.index:
    df_bigram_prob.loc[w,:] = df_bigram_prob.loc[w,:]/df_word_cnts.loc[0,w]
df_bigram_prob
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
      <th>i</th>
      <th>want</th>
      <th>to</th>
      <th>eat</th>
      <th>chinese</th>
      <th>food</th>
      <th>lunch</th>
      <th>spend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>i</th>
      <td>0.001974</td>
      <td>0.32649</td>
      <td>0.000000</td>
      <td>0.003553</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000790</td>
    </tr>
    <tr>
      <th>want</th>
      <td>0.002157</td>
      <td>0.00000</td>
      <td>0.655879</td>
      <td>0.001079</td>
      <td>0.006472</td>
      <td>0.006472</td>
      <td>0.005394</td>
      <td>0.001079</td>
    </tr>
    <tr>
      <th>to</th>
      <td>0.000827</td>
      <td>0.00000</td>
      <td>0.001655</td>
      <td>0.283823</td>
      <td>0.000827</td>
      <td>0.000000</td>
      <td>0.002482</td>
      <td>0.087298</td>
    </tr>
    <tr>
      <th>eat</th>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.002681</td>
      <td>0.000000</td>
      <td>0.021448</td>
      <td>0.002681</td>
      <td>0.056300</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>chinese</th>
      <td>0.006329</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.518987</td>
      <td>0.006329</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>food</th>
      <td>0.013724</td>
      <td>0.00000</td>
      <td>0.013724</td>
      <td>0.000000</td>
      <td>0.000915</td>
      <td>0.003660</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>lunch</th>
      <td>0.005865</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002933</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>spend</th>
      <td>0.003597</td>
      <td>0.00000</td>
      <td>0.003597</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



請根據已給的機率與所計算出的機率(df_bigram_prob), 試著判斷下列兩個句子哪個較為合理

s1 = “i want english food”

s2 = "want i english food"


```python
opd={}
opd["P(i|_start_)"] = 0.25
opd["P(want|i)"] = df_bigram_prob.loc['i','want']
opd["P(english|want)"]=0.0011
opd["P(food|english)"]=0.5
opd["P(_end_|food)"]=0.68
opd["P(want|_start_)"]=0.25
opd["P(i|want)"]=df_bigram_prob.loc['want','i']
opd["P(english|i)"] = 0.0011
```


```python
'''
###<your answer>###
'''
s1 = '_start_ i want english food _end_'
sp = s1.split(" ")
P_s = 1.0

output_str = []
for i in range(0,len(sp)-1):
    output_str.append(f"P({sp[i+1]}|{sp[i]})")
    P_s*=round(opd[output_str[-1]],6)
print(f'P(s1) = {"*".join(output_str)} = {P_s:.13f}')

s2 = '_start_ want i english food _end_'
sp = s2.split(" ")
P_s = 1.0
output_str = []
for i in range(0,len(sp)-1):
    output_str.append(f"P({sp[i+1]}|{sp[i]})")
    P_s*=round(opd[output_str[-1]],6)
print(f'P(s2) = {"*".join(output_str)} = {P_s:.13f}')
```

    P(s1) = P(i|_start_)*P(want|i)*P(english|want)*P(food|english)*P(_end_|food) = 0.0000305268150
    P(s2) = P(want|_start_)*P(i|want)*P(english|i)*P(food|english)*P(_end_|food) = 0.0000002016795


s1 較為合理


```python

```
