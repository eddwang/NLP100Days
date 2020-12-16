### 作業目的:了解Ensemble中的Blending與Stacking

### Question: 請描述Blending與Stacking的差異

Answer: Stacking 對train data使用了 k-fold切分去使用在不同的分類器上，然後用 test data 產生預測作為新的特徵，達成合併不同模型的作用，Blending 對 train data採用holdout切分(不包含 test data)，用不同的分類器訓練沒有holdout的部份，然後用holdout產生新的特徵，Blending 在 test data 上有做和 Stacking 相同的行為


```python

```
