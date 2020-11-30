# 作業目標: 利用正規表達式達到預期配對
本次作業將以互動式網站[Regex101](https://regex101.com/)來做練習，請將所需配對文本複製貼上到Regex101的**TEST STRING**區塊來做練習

### HW1: 電話號碼配對

抓出在電話號碼的所在地區以及號碼

```
ex: 02-33334444 --> 配對02, 33334444
```


**所需配對文本:**
```
02-27208889
04-2220-3585
(06)-2991111
(07)799-5678
```

**應配對出的結果為**
```
02, 27208889
04, 22203585
06, 2991111
07, 7995678
```

\d+

### HW2: 身分證字號配對
請配對出找出桃園(H), 台南(D), 嘉義(Q)中為男生的身分證字號(數字為1開頭)

**所需配對文本:**
```
A121040176
L186856359
Z127598010
I114537095
D279884447
L186834359
D243736345
I114537095
Q146110887
D187217314
I114537095
Q243556025
Z127598010
H250077453
Q188367037
```

**應配對出的結果為**
```
Q146110887
D187217314
Q188367037
```

[HQD]1/d+

### HW3: 電子郵件配對
請抓出非gmail的電子郵件

**所需配對文本:**
```
foobar@gmail.com
NoOneCareMe@gmail.com
SaveTheWorld@hotmail.com
zzzGroup@yahoo.com
eagle1963@gmail.com
maythefourthwithyiu@starwars.com
```

**應配對出的結果為**
```
SaveTheWorld@hotmail.com
zzzGroup@yahoo.com
maythefourthwithyiu@starwars.com
```


.+@(?!gmail)\w+\.\w+

### HW4: HTML格式配對

請抓出<TAG>當中的Tag就好，裡面的屬性請排除。

```
ex: <p class='test'> --> 抓出 p
```

**所需配對文本:**
```
<h1>This is a header 1</h1>
<a>This is a hyperlink</a>
<div class='test'>This is a text block</div>
<a href="https://regexisfun.com.tw/">Learning Regular Expression</a>
```

**應配對出的結果為**
```
h1
a
div
a
```

(?<=<)(\w+)

### HW5: 特定檔案名稱與格式配對

在所有檔案中，抓出屬於 gif 或 jpg 的檔名。


**所需配對文本:**
```
.bash_profile
workShop.ai
file_folderName_num.jpg
favicon.png
IMG_002.png
IMG_003.gif
qoo.jpg.tmp
index.html
foobar.bmp
foobar.jpg
account.html
access.lock
```

**應配對出的結果為**
```
IMG_003.gif
file_folderName_num.jpg
foobar.jpg
```

(.+\.gif|.+\.jpg)\n

### HW6: URL配對

請抓出 Url 中的協定方式, 網址, 與埠

```
ex: Https://localhost:4200/ --> 抓取 Https, localhost, 4200
```

**所需配對文本:**
```
ftp://file_server.com:21/account/customers.xml
https://hengxiuxu.blogspot.tw/
file://localhost:4200
https://s3cur3-server.com:9999/
```

**應配對出的結果為**
```
ftp, file_server, 21
https, hengxiuxu.blogspot.tw
file, localhost, 4200
https, s3cur3-server.com, 9999
```

\w+(?=://)|(?<=//)[\w\.\-]+|(?<=:)\d+


```python

```
