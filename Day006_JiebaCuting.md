## 作業目標: 使用Jieba進行各項的斷詞操作

這份作業我們會使用中文飯店評論資料集來作為斷詞練習。 [資料集:ChnSentiCorp_htl_all](https://github.com/SophonPlus/ChineseNlpCorpus)

### 讀入文本資料


```python
import pandas as pd

# hint: 可利用pandas讀取CSV
###<your code>###
pd_corpus=pd.read_csv("ChnSentiCorp_htl_all.csv")
pd_corpus.head(5)
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
      <th>label</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>距离川沙公路较近,但是公交指示不对,如果是"蔡陆线"的话,会非常麻烦.建议用别的路线.房间较...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>早餐太差，无论去多少人，那边也不加食品的。酒店应该重视一下这个问题了。房间本身很好。</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>宾馆在小街道上，不大好找，但还好北京热心同胞很多~宾馆设施跟介绍的差不多，房间很小，确实挺小...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>CBD中心,周围没什么店铺,说5星有点勉强.不知道为什么卫生间没有电吹风</td>
    </tr>
  </tbody>
</table>
</div>




```python
#確認所有留言,正評價(label=1)與負評價數量(label=0)
###<your code>###
pd_positive = pd_corpus[pd_corpus['label']==1]
pd_negative=pd_corpus[pd_corpus['label']==0]
print(f'Total: {len(pd_corpus)}, Positive: {len(pd_positive)}, Negative: {len(pd_negative)}')
```

    Total: 7766, Positive: 5322, Negative: 2444



```python

```


```python
#移除缺失值
###<your code>###
pd_corpus=pd_corpus[~pd_corpus['review'].isna()]
pd_corpus=pd_corpus[~pd_corpus['label'].isna()]
pd_corpus
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
      <th>label</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>距离川沙公路较近,但是公交指示不对,如果是"蔡陆线"的话,会非常麻烦.建议用别的路线.房间较...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>早餐太差，无论去多少人，那边也不加食品的。酒店应该重视一下这个问题了。房间本身很好。</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>宾馆在小街道上，不大好找，但还好北京热心同胞很多~宾馆设施跟介绍的差不多，房间很小，确实挺小...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>CBD中心,周围没什么店铺,说5星有点勉强.不知道为什么卫生间没有电吹风</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7761</th>
      <td>0</td>
      <td>尼斯酒店的几大特点：噪音大、环境差、配置低、服务效率低。如：1、隔壁歌厅的声音闹至午夜3点许...</td>
    </tr>
    <tr>
      <th>7762</th>
      <td>0</td>
      <td>盐城来了很多次，第一次住盐阜宾馆，我的确很失望整个墙壁黑咕隆咚的，好像被烟熏过一样家具非常的...</td>
    </tr>
    <tr>
      <th>7763</th>
      <td>0</td>
      <td>看照片觉得还挺不错的，又是4星级的，但入住以后除了后悔没有别的，房间挺大但空空的，早餐是有但...</td>
    </tr>
    <tr>
      <th>7764</th>
      <td>0</td>
      <td>我们去盐城的时候那里的最低气温只有4度，晚上冷得要死，居然还不开空调，投诉到酒店客房部，得到...</td>
    </tr>
    <tr>
      <th>7765</th>
      <td>0</td>
      <td>说实在的我很失望，之前看了其他人的点评后觉得还可以才去的，结果让我们大跌眼镜。我想这家酒店以...</td>
    </tr>
  </tbody>
</table>
<p>7765 rows × 2 columns</p>
</div>



### 建構結巴斷詞Function

建構可將讀入的Pandas DataFrame的文本資料，外加一欄名為`cut`的review欄位斷詞結果


```python
import jieba
import logging
jieba.setLogLevel(logging.INFO)
```


```python
class JiebaCutingClass(object):
    '''Class to use jeiba to parse corpus from dataframe and cut the corpus
    
    Parameters
    -----------
    key_to_cut: str
        the dataframe key to parse the sentence for jieba cutting
    dic: str
        the dictionary to use for jieba, default is None (use default dictionary)
    userdict: str
        the user defined dictionary to use for jieba, default is None
    '''
    
    def __init__(self, key_to_cut:str, dic:str=None, userdict:str=None):
        
        if dic is not None:
            jieba.set_dictionary(dic)
        
        if userdict is not None:
            jieba.load_userdict(userdict)
        
        self.key_to_cut = key_to_cut
        jieba.enable_paddle()
        #將paddle 功能開啟
        ###<your code>###
        
    @staticmethod
    def cut_single_sentence(sentence, use_paddle=False, use_full=False, use_search=False):
        
        out=[]
        if use_search:
            # hint:使用收尋引擎模式進行斷詞
            ###<your code>###
            out=list(jieba.cut_for_search(sentence))
        else:
            # hint:非收尋引擎模式斷詞，請注意有精確模式、全模式與paddle模式
            ###<your code>###
            out=list(jieba.cut(sentence, use_paddle=True, cut_all=True))          
        return out
            
    
    def cut_corpus(self, corpus: pd.DataFrame, mode: str) -> pd.DataFrame:
        '''Method to read and cut sentence from dataframe and append another column named cut
        
        Paremeters
        --------------
        corpus: pd.DataFrame
            Input corpus in dataframe
        mode: str
            Jieba mode to be used
        
        Return
        ----------------
        out: pd.Dataframe
            Output corpus in dataframe
        '''
        
        # checking valid mode
        if mode not in ['paddle', 'full', 'precise', 'search']:
            raise TypeError(f'only support `paddle`, `full`, `precise`, and `search` mode, but get {mode}')
            
        # cut the corpus based on mode
        # hint: 根據mode來呼叫相對應的斷詞method
        if mode == 'paddle':
            ###<your code>###           
            out=self._paddle_cut(corpus)
        elif mode == 'full':
            ###<your code>###
            out=self._full_cut(corpus)
        elif mode == 'precise':
            ###<your code>###
            out=self._precise_cut(corpus)
        elif mode == 'search':
            ###<your code>###
            out=self._search_cut(corpus)

        return out
    
    def _paddle_cut(self, corpus):
        '''paddle mode
        '''
        #enable paddle
        #hint:先啟用paddle mode
        ### <your code> ###
        
        out = []
        # hint:將讀進的文本進行斷詞，並將結果append回out的陣列中
        for single_review in corpus[self.key_to_cut]:
            ###<your code>###
            out.append(self.cut_single_sentence(single_review,use_paddle=True))
        corpus['cut'] = out
        
        return corpus
    
    def _full_cut(self, corpus):
        '''full mode
        '''
        
        out = []
        # hint:將讀進的文本進行斷詞，並將結果append回out的陣列中
        for single_review in corpus[self.key_to_cut]:
            ###<your code>###
            out.append(self.cut_single_sentence(single_review,use_full=True))
        corpus['cut'] = out
        
        return corpus
    
    def _precise_cut(self, corpus):
        '''precise mode
        '''
        
        out = []
        # hint:將讀進的文本進行斷詞，並將結果append回out的陣列中
        for single_review in corpus[self.key_to_cut]:
            ###<your code>###
            out.append(self.cut_single_sentence(single_review))
        corpus['cut'] = out
        
        return corpus
    
    def _search_cut(self, corpus):
        '''search mode
        '''
            
        out = []
        # hint:將讀進的文本進行斷詞，並將結果append回out的陣列中
        for single_review in corpus[self.key_to_cut]:
            ###<your code>###
            out.append(self.cut_single_sentence(single_review,use_search=True))
        corpus['cut'] = out
        
        return corpus
```

### 使用建構好的斷詞物件對文本進行斷詞


```python
jibCut=JiebaCutingClass(key_to_cut="review")
```

    Paddle enabled successfully......



```python
pd_cut=jibCut.cut_corpus(pd_corpus.head(50),mode='precise')
```

    /home/yau/anaconda3/envs/py37/lib/python3.7/site-packages/ipykernel_launcher.py:117: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy



```python
### 請使用精確模式與預設字典對文本進行斷詞

# hint:請先實例化JiebaCutingClass，再進行斷詞
###<your code>###
###<your code>### #為了避免處理時間過久, 這裡我們只使用前50個進行斷詞

pd_cut.head()
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
      <th>label</th>
      <th>review</th>
      <th>cut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>距离川沙公路较近,但是公交指示不对,如果是"蔡陆线"的话,会非常麻烦.建议用别的路线.房间较...</td>
      <td>[距离, 川沙公路, 较, 近, ,, 但是, 公交, 指示, 不对, ,, 如果, 是, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!</td>
      <td>[商务, 大床, 房, ，, 房间, 很大, ，床, 有, 2M, 宽, ，, 整体, 感觉...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>早餐太差，无论去多少人，那边也不加食品的。酒店应该重视一下这个问题了。房间本身很好。</td>
      <td>[早餐, 太, 差，, 无, 论, 去, 多少, 人, ，, 那边, 也, 不, 加, 食品...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>宾馆在小街道上，不大好找，但还好北京热心同胞很多~宾馆设施跟介绍的差不多，房间很小，确实挺小...</td>
      <td>[宾馆, 在, 小街道, 上, ，不大好找, ，, 但, 还好, 北京, 热心, 同胞, 很...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>CBD中心,周围没什么店铺,说5星有点勉强.不知道为什么卫生间没有电吹风</td>
      <td>[CBD中心, ,, 周围, 没, 什么, 店铺, ,, 说, 5, 星, 有点, 勉强, ...</td>
    </tr>
  </tbody>
</table>
</div>



### 載入繁中字典為指定的字詞進行斷詞


```python
test_string = '我愛cupoy自然語言處理馬拉松課程'
jieba_cut = JiebaCutingClass(key_to_cut='', dic='./dict.txt.big')


out_string =jieba_cut.cut_single_sentence(test_string,use_paddle=True) ###<your code>### #paddle 模式
print(f'Paddle模式: {[string for string in out_string]}')

out_string = jieba_cut.cut_single_sentence(test_string,use_full=True) ###<your code>### #全模式
print(f'全模式: {[string for string in out_string]}')

out_string = jieba_cut.cut_single_sentence(test_string,use_search=True)###<your code>### #搜尋模式
print(f'搜尋模式: {[string for string in out_string]}')

out_string = jieba_cut.cut_single_sentence(test_string)###<your code>### #精確模式
print(f'精確模式: {[string for string in out_string]}')
```

    Paddle enabled successfully......


    Paddle模式: ['我', '愛', 'cupoy', '自然', '語言', '處理', '馬拉松', '課程']
    全模式: ['我', '愛', 'cupoy', '自然', '語言', '處理', '馬拉松', '課程']
    搜尋模式: ['我', '愛', 'cupoy', '自然', '語言', '自然語言', '處理', '馬拉', '馬拉松', '課程']
    精確模式: ['我', '愛', 'cupoy', '自然', '語言', '處理', '馬拉松', '課程']



```python

```
