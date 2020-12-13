### 作業目的:了解決策樹的節點分支依據
本次作業可參考簡報中的延伸閱讀[訊息增益](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda)部分

#### Question:
若你是決策樹，下列兩種分類狀況(a,b)，你會選擇哪種做分類？為什麼？

<img src='hw_1.png' style='width:500px'>




```python
import math
Dp=-(0.5*math.log2(0.5)+0.5*math.log2(0.5))
Dal=-((15)/(20)*math.log2((15)/(20))+(5)/(20)*math.log2((5)/(20)))
Dar=-((5)/(20)*math.log2((5)/(20))+(15)/(20)*math.log2((15)/(20)))

Dbl=-((15)/(35)*math.log2((15)/(35))+(10)/(35)*math.log2((20)/(35)))
Dbr=-((5)/(5)*math.log2((5)/(5))+(0)/(5)*math.log2((0+1e-10)/(5)))

print("(a)",Dp-0.5*Dal-0.5*Dar)
print("(b)",Dp-0.875*Dbl-0.125*Dbr)
```

    (a) 0.1887218755408671
    (b) 0.339764111484431


### Answer:

(b)的情況以entropy計算時增益大過(a)因此應該選(b)


### 閱讀作業

決策樹根據計算分割準則的不同(ex: Entropy, Gini, Gain ratio)，可分為ID3, C4.5, CART樹的算法，請同學閱讀下列文章，來更加了解決策樹的算法。

[決策樹(ID3, C4.5, CART)](https://blog.csdn.net/u010089444/article/details/53241218)


```python

```
