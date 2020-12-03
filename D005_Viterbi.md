## 作業目標: 了解斷詞演算法的背後計算

### 根據課程講述的內容, 請計算出下列剩餘所有情況的
若有一個人連續觀察到三天水草都是乾燥的(Dry), 則這三天的天氣機率為何？(可參考講義第13頁)
(Hint: 共有8種可能機率)

```python
states = ('sunny', 'rainy')
observations = ('dry', 'damp', 'soggy')
start_probability = {'sunny': 0.4, 'rainy': 0.6}
transition_probability = {'sunny':{'sunny':0.6, 'rainy':0.4},
                          'rainy': {'sunny':0.3, 'rainy':0.7}}
emission_probatility = {'sunny': {'dry':0.6, 'damp':0.3, 'soggy':0.1},
                        'rainy': {'dry':0.1, 'damp':0.4, 'soggy':0.5}}
```

```
觀察狀態 = 'dry', 'dry', 'dry'
Sunny, Sunny, Sunny: 0.4*(0.6)*0.6*(0.6)*0.6*(0.6) = 0.031104
Rainy, Sunny, Sunny: 0.6*(0.1)*0.3*(0.6)*0.6*(0.6) = 0.003888
Sunny, Rainy, Sunny: 0.4*(0.6)*0.4*(0.1)*0.3*(0.6) = 0.001728
Sunny, Sunny, Rainy: 0.4*(0.6)*0.6*(0.6)*0.4*(0.1) = 0.003456
Rainy, Rainy, Sunny: 0.6*(0.1)*0.7*(0.1)*0.3*(0.6) = 0.000756
Rainy, Sunny, Rainy: 0.6*(0.1)*0.3*(0.6)*0.4*(0.1) = 0.000432
Sunny, Rainy, Rainy: 0.4*(0.6)*0.4*(0.1)*0.7*(0.1) = 0.000672
Rainy, Rainy, Rainy: 0.6*(0.1)*0.7*(0.1)*0.7*(0.1) = 0.000294

最大機率為: Sunny, Sunny, Sunny
```

### 根據上述條件, 寫出Viterbi應用程式


```python
observations = ('dry', 'dry', 'dry') #實際上觀察到的狀態為dry, dry, dry
states = ('sunny', 'rainy')
start_probability = {'sunny': 0.4, 'rainy': 0.6}
transition_probability = {'sunny':{'sunny':0.6, 'rainy':0.4},
                          'rainy': {'sunny':0.3, 'rainy':0.7}}
emission_probatility = {'sunny': {'dry':0.6, 'damp':0.3, 'soggy':0.1},
                        'rainy': {'dry':0.1, 'damp':0.4, 'soggy':0.5}}
```


```python
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        ###<your codes>###
        # 初始狀態，取得第一個觀察值
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
        
    # Run Viterbi for t > 0
    for t in range(1,len(obs)):
        ###<your codes>###
        # 開始計算剩下的觀察值
        V.append({})
        newpath = {}
        
        for cur_state in states:
            # 每個狀態的最佳路徑
            (prob, state) = max([(V[t-1][pre_state] * trans_p[pre_state][cur_state] * emit_p[cur_state][obs[t]], pre_state) for pre_state in states])
            V[t][cur_state] = prob
            newpath[cur_state] = path[state] + [cur_state]


        # Don't need to remember the old paths
        path = newpath
    
    # 取出最終結果的路徑與機率值
    (prob, state) = max([(V[len(obs) - 1][final_state], final_state) for final_state in states])
    return (prob, path[state])
```


```python
result = viterbi(observations,
                 states,
                 start_probability,
                 transition_probability,
                 emission_probatility)
```


```python
result
```




    (0.031103999999999993, ['sunny', 'sunny', 'sunny'])




```python

```
