# 作業目標：運用課程所學，操作字串達到預期輸出

---

### isnumeric(), isdigit(), isdecimal() 各有幾個


```python
test_string = ['5.9', '30', '½', '³', '⑬']
```


```python
def spam(s, isnumeric_count, isdigit_count, isdecimal_count):
    for si in s:
        isnumeric_count += si.isnumeric()
        isdigit_count += si.isdigit()
        isdecimal_count += si.isdecimal()
    return isnumeric_count, isdigit_count, isdecimal_count
```


```python
isnumeric_count, isdigit_count, isdecimal_count=spam(test_string,0,0,0)
```


```python
print('isnumeric_count: {}'.format(isnumeric_count))     
print('isdigit_count: {}'.format(isdigit_count))     
print('isdecimal_count: {}'.format(isdecimal_count))  
```

    isnumeric_count: 4
    isdigit_count: 2
    isdecimal_count: 1


---

## 運用formatting 技巧 output
    * Accuracy: 98.13%, Recall: 94.88%, Precision: 96.29%


```python
accuracy = 98.129393
recall =   94.879583
precision =96.294821
```


```python
print("Accuracy: {0:.2f}%, Recall:{1:.2f}%, Precision: {2:.2f}%".format(accuracy,recall,precision))
```

    Accuracy: 98.13%, Recall:94.88%, Precision: 96.29%



```python
accuracy = 0.98129393
recall =   0.94879583
precision =0.96294821
```


```python
print("Accuracy: {0:.2f}%, Recall:{1:.2f}%, Precision: {2:.2f}%".format(accuracy*1e2,recall*1e2,precision*1e2))
```

    Accuracy: 98.13%, Recall:94.88%, Precision: 96.29%


---

## 依照只是轉換number output format


```python
number = 3.1415926
```

#### 轉換為科學符號表示法 (小數點後兩位)


```python
print('{0:.2e}'.format(number))
```

    3.14e+00


#### 轉換為%


```python
print('{0:.2%}'.format(number))
```

    314.16%


#### 補零成為3.14159300


```python
print('{0:0<10f}'.format(number))
```

    3.14159300



```python

```
