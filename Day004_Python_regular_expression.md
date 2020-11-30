# 作業目標: 使用python正規表達式對資料進行清洗處理

這份作業我們會使用詐欺郵件的文本資料來作為清洗與處理的操作。
[資料集](https://www.kaggle.com/rtatman/fraudulent-email-corpus/data#)

### 讀入資料文本
因原始文本較大，先使用部份擷取的**sample_emails.txt**來進行練習


```python
#讀取文本資料
with open('sample_emails.txt', 'r', encoding="utf8", errors='ignore') as f:
    sample_corpus = f.read()
```


```python
sample_corpus
```




    'From r  Wed Oct 30 21:41:56 2002\nReturn-Path: <james_ngola2002@maktoob.com>\nX-Sieve: cmu-sieve 2.0\nReturn-Path: <james_ngola2002@maktoob.com>\nMessage-Id: <200210310241.g9V2fNm6028281@cs.CU>\nFrom: "MR. JAMES NGOLA." <james_ngola2002@maktoob.com>\nReply-To: james_ngola2002@maktoob.com\nTo: webmaster@aclweb.org\nDate: Thu, 31 Oct 2002 02:38:20 +0000\nSubject: URGENT BUSINESS ASSISTANCE AND PARTNERSHIP\nX-Mailer: Microsoft Outlook Express 5.00.2919.6900 DM\nMIME-Version: 1.0\nContent-Type: text/plain; charset="us-ascii"\nContent-Transfer-Encoding: 8bit\nX-MIME-Autoconverted: from quoted-printable to 8bit by sideshowmel.si.UM id g9V2foW24311\nStatus: O\n\nFROM:MR. JAMES NGOLA.\nCONFIDENTIAL TEL: 233-27-587908.\nE-MAIL: (james_ngola2002@maktoob.com).\n\nURGENT BUSINESS ASSISTANCE AND PARTNERSHIP.\n\n\nDEAR FRIEND,\n\nI AM ( DR.) JAMES NGOLA, THE PERSONAL ASSISTANCE TO THE LATE CONGOLESE (PRESIDENT LAURENT KABILA) WHO WAS ASSASSINATED BY HIS BODY GUARD ON 16TH JAN. 2001.\n\n\nTHE INCIDENT OCCURRED IN OUR PRESENCE WHILE WE WERE HOLDING MEETING WITH HIS EXCELLENCY OVER THE FINANCIAL RETURNS FROM THE DIAMOND SALES IN THE AREAS CONTROLLED BY (D.R.C.) DEMOCRATIC REPUBLIC OF CONGO FORCES AND THEIR FOREIGN ALLIES ANGOLA AND ZIMBABWE, HAVING RECEIVED THE PREVIOUS DAY (USD$100M) ONE HUNDRED MILLION UNITED STATES DOLLARS, CASH IN THREE DIPLOMATIC BOXES ROUTED THROUGH ZIMBABWE.\n\nMY PURPOSE OF WRITING YOU THIS LETTER IS TO SOLICIT FOR YOUR ASSISTANCE AS TO BE A COVER TO THE FUND AND ALSO COLLABORATION IN MOVING THE SAID FUND INTO YOUR BANK ACCOUNT THE SUM OF (USD$25M) TWENTY FIVE MILLION UNITED STATES DOLLARS ONLY, WHICH I DEPOSITED WITH A SECURITY COMPANY IN GHANA, IN A DIPLOMATIC BOX AS GOLDS WORTH (USD$25M) TWENTY FIVE MILLION UNITED STATES DOLLARS ONLY FOR SAFE KEEPING IN A SECURITY VAULT FOR ANY FURTHER INVESTMENT PERHAPS IN YOUR COUNTRY. \n\nYOU WERE INTRODUCED TO ME BY A RELIABLE FRIEND OF MINE WHO IS A TRAVELLER,AND ALSO A MEMBER OF CHAMBER OF COMMERCE AS A RELIABLE AND TRUSTWORTHY PERSON WHOM I CAN RELY ON AS FOREIGN PARTNER, EVEN THOUGH THE NATURE OF THE TRANSACTION WAS NOT REVEALED TO HIM FOR SECURITY REASONS.\n\n\nTHE (USD$25M) WAS PART OF A PROCEEDS FROM DIAMOND TRADE MEANT FOR THE LATE PRESIDENT LAURENT KABILA WHICH WAS DELIVERED THROUGH ZIMBABWE IN DIPLOMATIC BOXES. THE BOXES WERE KEPT UNDER MY CUSTODY BEFORE THE SAD EVENT THAT TOOK THE LIFE OF (MR. PRESIDENT).THE CONFUSION THAT ENSUED AFTER THE ASSASSINATION AND THE SPORADIC SHOOTING AMONG THE FACTIONS, I HAVE TO RUN AWAY FROM THE COUNTRY FOR MY DEAR LIFE AS I AM NOT A SOLDIER BUT A CIVIL SERVANT I CROSSED RIVER CONGO TO OTHER SIDE OF CONGO LIBREVILLE FROM THERE I MOVED TO THE THIRD COUNTRY GHANA WHERE I AM PRESENTLY TAKING REFUGE. \n\nAS A MATTER OF FACT, WHAT I URGENTLY NEEDED FROM YOU IS YOUR ASSISTANCE IN MOVING THIS MONEY INTO YOUR ACCOUNT IN YOUR COUNTRY FOR INVESTMENT WITHOUT RAISING EYEBROW. FOR YOUR ASSISTANCE I WILL GIVE YOU 20% OF THE TOTAL SUM AS YOUR OWN SHARE WHEN THE MONEY GETS TO YOUR ACCOUNT, WHILE 75% WILL BE FOR ME, OF WHICH WITH YOUR KIND ADVICE I HOPE TO INVEST IN PROFITABLE VENTURE IN YOUR COUNTRY IN OTHER TO SETTLE DOWN FOR MEANINGFUL LIFE, AS I AM TIRED OF LIVING IN A WAR ENVIRONMENT. \n\nTHE REMAINING 5% WILL BE USED TO OFFSET ANY COST INCURRED IN THE CAUSE OF MOVING THE MONEY TO YOUR ACCOUNT. IF THE PROPOSAL IS ACCEPTABLE TO YOU PLEASE CONTACT ME IMMEDIATELY THROUGH THE ABOVE TELEPHONE AND E-MAIL, TO ENABLE ME ARRANGE FACE TO FACE MEETING WITH YOU IN GHANA FOR THE CLEARANCE OF THE FUNDS BEFORE TRANSFRING IT TO YOUR BANK ACCOUNT AS SEEING IS BELIEVING. \n\nFINALLY, IT IS IMPORTANT ALSO THAT I LET YOU UNDERSTAND THAT THERE IS NO RISK INVOLVED WHATSOEVER AS THE MONEY HAD NO RECORD IN KINSHASA FOR IT WAS MEANT FOR THE PERSONAL USE OF (MR. PRESIDEND ) BEFORE THE NEFARIOUS INCIDENT OCCURRED, AND ALSO I HAVE ALL THE NECESSARY DOCUMENTS AS REGARDS TO THE FUNDS INCLUDING THE (CERTIFICATE OF DEPOSIT), AS I AM THE DEPOSITOR OF THE CONSIGNMENT.\n\n\nLOOKING FORWARD TO YOUR URGENT RESPONSE.\n\nYOUR SINCERELY,\n\nMR. JAMES NGOLA. \n\n\n\n\n\n\n\n\n\n\nFrom r  Thu Oct 31 08:11:39 2002\nReturn-Path: <bensul2004nng@spinfinder.com>\nX-Sieve: cmu-sieve 2.0\nReturn-Path: <bensul2004nng@spinfinder.com>\nMessage-Id: <200210311310.g9VDANt24674@bloodwork.mr.itd.UM>\nFrom: "Mr. Ben Suleman" <bensul2004nng@spinfinder.com>\nDate: Thu, 31 Oct 2002 05:10:00\nTo: R@M\nSubject: URGENT ASSISTANCE /RELATIONSHIP (P)\nMIME-Version: 1.0\nContent-Type: text/plain;charset="iso-8859-1"\nContent-Transfer-Encoding: 7bit\nStatus: O\n\nDear Friend,\n\nI am Mr. Ben Suleman a custom officer and work as Assistant controller of the Customs and Excise department Of the Federal Ministry of Internal Affairs stationed at the Murtala Mohammed International Airport, Ikeja, Lagos-Nigeria.\n\nAfter the sudden death of the former Head of state of Nigeria General Sanni Abacha on June 8th 1998 his aides and immediate members of his family were arrested while trying to escape from Nigeria in a Chartered jet to Saudi Arabia with 6 trunk boxes Marked "Diplomatic Baggage". Acting on a tip-off as they attempted to board the Air Craft,my officials carried out a thorough search on the air craft and discovered that the 6 trunk boxes contained foreign currencies amounting to US$197,570,000.00(One Hundred and  Ninety-Seven Million Five Hundred Seventy Thousand United States Dollars).\n\nI declared only (5) five boxes to the government and withheld one (1) in my custody containing the sum of (US$30,000,000.00) Thirty Million United States Dollars Only, which has been disguised to prevent their being discovered during transportation process.Due to several media reports on the late head of state about all the money him and his co-government officials stole from our government treasury amounting\nto US$55 Billion Dollars (ref:ngrguardiannews.com) of July 2nd 1999. Even the London times of July 1998 reported that General Abacha has over US$3.Billion dollars in one account overseas. We decided to conceal this one (1)box till the situation is calm and quite on the issue. The box was thus deposited with a security company here in Nigeria and tagged as "Precious Stones and Jewellry" in other that its\ncontent will not be discovered. Now that all is calm, we (myself and two of my colleagues in the operations team) are now ready to move this box out of the country through a diplomatic arrangement which is the safest means. \n\nHowever as government officials the Civil Service Code of Conduct does not allow us by law to operate any foreign account or own foreign investment and the amount of money that can be found in our account\ncannot be more than our salary on the average, thus our handicapp and our need for your assistance to help collect and keep safely in your account this money.\n\nTherefore we want you to assist us in moving this money out of Nigeria. We shall definitely compensate you handsomely for the assistance. We can do this by instructing the Security Company here in Nigeria to\nmove the consignment to their affiliate branch office outside Nigeria through diplomatic means and the consignment will be termed as Precious Stones and Jewelleries" which you bought during your visit to Nigeria and is being transfered to your country from here for safe keeping. Then we can arrange to meet at the destination country to take the delivery of the consignment. You will thereafter open an account there and lodge the Money there and gradually instruct remittance to your Country. \n\nThis business is 100% risk free for you so please treat this matter with utmost confidentiality .If you indicate your interest to assist us please just e-mail me for more Explanation on how we plan to execute the transaction.\n\nExpecting your response urgently.\n\nBest regards,\n\nMr. Ben Suleman\n\nFrom r  Thu Oct 31 17:27:16 2002\nReturn-Path: <obong_715@epatra.com>\nX-Sieve: cmu-sieve 2.0\nReturn-Path: <obong_715@epatra.com>\nMessage-Id: <200210312227.g9VMQvDj017948@bluewhale.cs.CU>\nFrom: "PRINCE OBONG ELEME" <obong_715@epatra.com>\nReply-To: obong_715@epatra.com\nTo: webmaster@aclweb.org\nDate: Thu, 31 Oct 2002 22:17:55 +0100\nSubject: GOOD DAY TO YOU\nX-Mailer: Microsoft Outlook Express 5.00.2919.6900DM\nMIME-Version: 1.0\nContent-Type: text/plain; charset="us-ascii"\nContent-Transfer-Encoding: 8bit\nX-MIME-Autoconverted: from quoted-printable to 8bit by sideshowmel.si.UM id g9VMRBW20642\nStatus: RO\n\nFROM HIS ROYAL MAJESTY (HRM) CROWN RULER OF ELEME KINGDOM \nCHIEF DANIEL ELEME, PHD, EZE 1 OF ELEME.E-MAIL \nADDRESS:obong_715@epatra.com  \n\nATTENTION:PRESIDENT,CEO Sir/ Madam. \n\nThis letter might surprise you because we have met\nneither in person nor by correspondence. But I believe\nit is one day that you got to know somebody either in\nphysical or through correspondence. \n\nI got your contact through discreet inquiry from the\nchambers of commerce and industry of your country on\nthe net, you and your organization were revealed as\nbeing quite astute in private entrepreneurship, one\nhas no doubt in your ability to handle a financialbusiness transaction. \n\nHowever, I am the first son of His Royal\nmajesty,Obong.D. Eleme , and the traditional Ruler of\nEleme Province in the oil producing area of River\nState of Nigeria. I am making this contact to you in\nrespect of US$60,000,000.00 (Sixty Million United\nState Dollars), which I inherited, from my latefather. \n\nThis money was accumulated from royalties paid to my\nfather as compensation by the oil firms located in our\narea as a result of oil presence on our land, which\nhamper agriculture, which is our major source oflivelihood. \n\nUnfortunately my father died from protracted\ndiabetes.But before his death he called my attention\nand informed me that he lodged some funds on a two\nboxes with a security firm with an open beneficiary\nstatus. The lodgment security code number was also\nrevealed to me, he then advised me to look for a\nreliable business partner abroad, that will assist me\nin investing the money in a lucrative business as a\nresult of economic instability in Nigeria. So this is\nthe main reason why I am contacting you for us to move\nthis money from the security firm to any Country of\nyour choice for investment purpose. \n\nSo I will like you to be the ultimate beneficiary, so\nthat the funds can be moved in your name and\nparticulars to any Country of your choice where it\nwill be claimed and invested. Hence my father have had\nintimated the security firm personnel that the\nbeneficiary of the box is his foreign partner whose\nparticulars will be forwarded to the firm when due. \n\nBut I will guide you Accordingly. As soon as the funds\nreach, I will then come over to meet you in person, so\nthat we can discuss physically on investment\npotentials. Based on this assistance my Family and I\nhave unanimously decided to give you 30% of the total\nmoney, 5% for Charity home, 10% for expenses, which\nmay arise during this transaction, Fax and phone bills\ninclusive. The balance of 55% you will invest andmanaged for my Family. \n\nI hereby guarantee you that this is not government\nmoney, it is not drug money and it is not money from\narms deal. Though you have to maintain high degree of\nconfidentiality on this matter. I will give more\ndetails about the proceedings of this transaction as\nsoon as I receive your favorable reply. \n\nPlease reply to my Email Address:obong_715@epatra.com\nI hope this will be the beginning of a prosperous\nrelationship between my family and your family. \n\nNevertheless if you are for any reason not interested,\nkindly inform me immediately so that I will look foranother contact. \n\nI am waiting for your quick response. \n\nYours faithfully, \n\nPrince Obong Abbot \n'



### 讀取寄件者資訊
觀察文本資料可以發現, 寄件者資訊都符合以下格式

`From: <收件者姓名> <收件者電子郵件>`


```python
import re
```


```python
#<your code>#
pattern = r'From:.*'
match = re.findall(pattern, sample_corpus)
```


```python
match
```




    ['From: "MR. JAMES NGOLA." <james_ngola2002@maktoob.com>',
     'From: "Mr. Ben Suleman" <bensul2004nng@spinfinder.com>',
     'From: "PRINCE OBONG ELEME" <obong_715@epatra.com>']



### 只讀取寄件者姓名


```python
#<your code>#
pattern = r'(?<=From: )\"[\w \.]+"'
match = re.findall(pattern, sample_corpus)
print("\n".join(match ))
```

    "MR. JAMES NGOLA."
    "Mr. Ben Suleman"
    "PRINCE OBONG ELEME"


### 只讀取寄件者電子信箱


```python
#<your code>#
pattern = r'From:[\s"\.\w]+\<(.+)?\>'
match = re.findall(pattern, sample_corpus)
print("\n".join(match ))
```

    james_ngola2002@maktoob.com
    bensul2004nng@spinfinder.com
    obong_715@epatra.com


### 只讀取電子信箱中的寄件機構資訊
ex: james_ngola2002@maktoob.com --> 取maktoob


```python
#<your code>#
pattern = r'From:[\s"\.\w]+\<.+@(.+)?\.\w+\>'
match = re.findall(pattern, sample_corpus)
print("\n".join(match ))
```

    maktoob
    spinfinder
    epatra


### 結合上面的配對方式, 將寄件者的帳號與機構訊返回
ex: james_ngola2002@maktoob.com --> [james_ngola2002, maktoob]


```python
#<your code>#
pattern = r'From:[\s"\.\w]+\<(.+)@(.+)?\.\w+\>'
match = re.findall(pattern, sample_corpus)
for s in match:
    print(", ".join(s))
```

    james_ngola2002, maktoob
    bensul2004nng, spinfinder
    obong_715, epatra


### 使用正規表達式對email資料進行處理
這裡我們會使用到python其他的套件協助處理(ex: pandas, email, etc)，這裡我們只需要專注在正規表達式上即可，其他的套件是方便我們整理與處理資料。

### 讀取與切分Email
讀入的email為一個長字串，利用正規表達式切割讀入的資料成一封一封的email，並將結果以list表示。

輸出: [email_1, email_2, email_3, ....]


```python
import re
import pandas as pd
import email

###讀取文本資料:fradulent_emails.txt###
#<your code>#
with open('all_emails.txt', 'r', encoding="utf8", errors='ignore') as f:
    all_emails = f.read()
    emails = ["From r"+s for s in all_emails.split("From r")]
    emails = emails[1:]
###切割讀入的資料成一封一封的email###
###我們可以使用list來儲存每一封email###
###注意！這裡請仔細觀察sample資料，看資料是如何切分不同email###
#<your code>#
len(emails) #查看有多少封email
```




    3977



### 從文本中擷取所有寄件者與收件者的姓名和地址


```python
emails_list = [] #創建空list來儲存所有email資訊

for mail in emails[:20]: #只取前20筆資料 (處理速度比較快)
    emails_dict = dict() #創建空字典儲存資訊
    ###取的寄件者姓名與地址###
    
    #Step1: 取的寄件者資訊 (hint: From:)
    #<your code>#
    m1 = re.search("(?<=From:).+\n",mail)
    #Step2: 取的姓名與地址 (hint: 要注意有時會有沒取到配對的情況)
    #<your code>#
    m2 = re.search(".+(?=<)",m1.group())
    m3 = re.search("(?<=<).+@.+\.\w\w\w(?=>)|[^<^\s.]+@.+",m1.group())

    #Step3: 將取得的姓名與地址存入字典中
    #<your code>#
    emails_dict['sender_name']=m2.group() if m2 != None else ""
    emails_dict['from']=m3.group() if m3 != None else ""
    
    
    ###取的收件者姓名與地址###
    #Step1: 取的寄件者資訊 (hint: To:)
    #<your code>#
    m4 = re.search("(?<=\nTo:).+",mail) 
    #Step2: 取的姓名與地址 (hint: 要注意有時會有沒取到配對的情況)
    #<your code>#
    m5 = re.search(".+(?=<)","" if m4 == None else m4.group())
    m6 = re.search("(?<=<).+@.+\.\w\w\w(?=>)|[^<^\s.]+@.+","" if m4 == None else m4.group())      
    #Step3: 將取得的姓名與地址存入字典中
    #<your code>#
    emails_dict['recipient_name']=m5.group() if m5 != None else ""
    emails_dict['to']=m6.group() if m6 != None else ""  

    ###取得信件日期###
    #Step1: 取得日期資訊 (hint: To:)
    #<your code>#
    m7 = re.search("(?<=Date).+",mail)
    #Step2: 取得詳細日期(只需取得DD MMM YYYY)
    #<your code>#
    m8 = re.search("\d+ \w+ \d+","" if m7 == None else m7.group())
    #Step3: 將取得的日期資訊存入字典中
    #<your code>#
    emails_dict['date'] = "" if m8 == None else m8.group()  
        
    ###取得信件主旨###
    #Step1: 取得主旨資訊 (hint: Subject:)
    #<your code>#
    m8 = re.search("(?<=Subject:).+",mail)
    #Step2: 移除不必要文字 (hint: Subject: )
    #<your code>#
    
    #Step3: 將取得的主旨存入字典中
    #<your code>#
    emails_dict['subject']=m8.group()
    
    ###取得信件內文###
    #這裡我們使用email package來取出email內文 (可以不需深究，本章節重點在正規表達式)
    try:
        full_email = email.message_from_string(mail)
        body = full_email.get_payload()
        emails_dict["email_body"] = body
    except:
        emails_dict["email_body"] = None
    
    ###將字典加入list###
    #<your code>#
    emails_list.append(emails_dict)
```


```python
#將處理結果轉化為dataframe
emails_df = pd.DataFrame(emails_list)
emails_df
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
      <th>sender_name</th>
      <th>from</th>
      <th>recipient_name</th>
      <th>to</th>
      <th>date</th>
      <th>subject</th>
      <th>email_body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>"MR. JAMES NGOLA."</td>
      <td>james_ngola2002@maktoob.com</td>
      <td></td>
      <td>webmaster@aclweb.org</td>
      <td>31 Oct 2002</td>
      <td>URGENT BUSINESS ASSISTANCE AND PARTNERSHIP</td>
      <td>FROM:MR. JAMES NGOLA.\nCONFIDENTIAL TEL: 233-2...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"Mr. Ben Suleman"</td>
      <td>bensul2004nng@spinfinder.com</td>
      <td></td>
      <td>R@M</td>
      <td>31 Oct 2002</td>
      <td>URGENT ASSISTANCE /RELATIONSHIP (P)</td>
      <td>Dear Friend,\n\nI am Mr. Ben Suleman a custom ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>"PRINCE OBONG ELEME"</td>
      <td>obong_715@epatra.com</td>
      <td></td>
      <td>webmaster@aclweb.org</td>
      <td>31 Oct 2002</td>
      <td>GOOD DAY TO YOU</td>
      <td>FROM HIS ROYAL MAJESTY (HRM) CROWN RULER OF EL...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"PRINCE OBONG ELEME"</td>
      <td>obong_715@epatra.com</td>
      <td></td>
      <td>webmaster@aclweb.org</td>
      <td>31 Oct 2002</td>
      <td>GOOD DAY TO YOU</td>
      <td>FROM HIS ROYAL MAJESTY (HRM) CROWN RULER OF EL...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>"Maryam Abacha"</td>
      <td>m_abacha03@www.com</td>
      <td></td>
      <td>R@M</td>
      <td>1 Nov 2002</td>
      <td>I Need Your Assistance.</td>
      <td>Dear sir, \n \nIt is with a heart full of hope...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kuta David</td>
      <td>davidkuta@postmark.net</td>
      <td></td>
      <td>davidkuta@yahoo.com</td>
      <td>02 Nov 2002</td>
      <td>Partnership</td>
      <td>ATTENTION:                                    ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>"Barrister tunde dosumu"</td>
      <td>tunde_dosumu@lycos.com</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Urgent Attention</td>
      <td>Dear Sir,\n\nI am Barrister Tunde Dosumu (SAN)...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>"William Drallo"</td>
      <td>william2244drallo@maktoob.com</td>
      <td></td>
      <td>webmaster@aclweb.org</td>
      <td>3 Nov 2002</td>
      <td>URGENT BUSINESS PRPOSAL</td>
      <td>FROM: WILLIAM DRALLO.\nCONFIDENTIAL TEL: 233-2...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>"MR USMAN ABDUL"</td>
      <td>abdul_817@rediffmail.com</td>
      <td></td>
      <td>R@M</td>
      <td>04 Nov 2002</td>
      <td>THANK YOU</td>
      <td>CHALLENGE SECURITIES LTD.\nLAGOS, NIGERIA\n\n\...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>"Tunde  Dosumu"</td>
      <td>barrister_td@lycos.com</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Urgent Assistance</td>
      <td>Dear Sir,\n\nI am Barrister Tunde Dosumu (SAN)...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MR TEMI JOHNSON</td>
      <td>temijohnson2@rediffmail.com</td>
      <td></td>
      <td>R@E</td>
      <td>6 Nov 2001</td>
      <td>IMPORTANT</td>
      <td>FROM: MR TEMI JOHNSON\nDEMOCRATIC REPUBLIC OF ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>"Dr.Sam jordan"</td>
      <td>sjordan@diplomats.com</td>
      <td></td>
      <td>R@M</td>
      <td>08 Nov 2002</td>
      <td>URGENT ASSISTANCE.</td>
      <td>FROM THE DESK OF DR.SAM JORDAN\nTHE MANAGER\nH...</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td>p_brown2@lawyer.com</td>
      <td></td>
      <td>webmaster@aclweb.org</td>
      <td>8 Nov 2002</td>
      <td>From: Barrister Peter Brown</td>
      <td>\nSENIOR ADVOCATE OF NIGERIA\nBARR. PETER BROW...</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>mic_k1@post.com</td>
      <td></td>
      <td>R@M</td>
      <td>11 Nov 2002</td>
      <td>MICHAEL</td>
      <td>From;Mr.Michael Kamah and Family,\n          J...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>"COL. MICHAEL BUNDU"</td>
      <td>mikebunduu1@rediffmail.com</td>
      <td></td>
      <td>R@E</td>
      <td>13 Nov 2001</td>
      <td>*****SPAM***** IMPORTANT</td>
      <td>FROM: COL. MICHAEL BUNDU. \nDEMOCRATIC REPUBLI...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>"MRS MARIAM ABACHA"</td>
      <td>elixwilliam@usa.com</td>
      <td></td>
      <td>webmaster@aclweb.org</td>
      <td>14 Nov 2002</td>
      <td>TRUST TRANSACTION</td>
      <td>Dear Sir,\n\nA Very Good day to you   \n\nI am...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>" DR. ANAYO AWKA "</td>
      <td>anayoawka@hotmail.com</td>
      <td></td>
      <td>webmaster@aclweb.org</td>
      <td>15 Nov 2002</td>
      <td>REQUEST FOR YOUR UNRESERVED ASSISTANCE</td>
      <td>FROM: DR. ANAYO AWKA BANK MANAGER \n(UNION BAN...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>" DR. ANAYO AWKA "</td>
      <td>anayoawka@hotmail.com</td>
      <td></td>
      <td>webmaster@aclweb.org</td>
      <td>15 Nov 2002</td>
      <td>REQUEST FOR YOUR UNRESERVED ASSISTANCE</td>
      <td>FROM: DR. ANAYO AWKA BANK MANAGER \n(UNION BAN...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>"Victor Aloma"</td>
      <td>victorloma@netscape.net</td>
      <td></td>
      <td>ntcir-listmem@newns.op.nii.ac.jp</td>
      <td>16 Nov 2002</td>
      <td>Urgent Assistance</td>
      <td>The Director,\n\n\n\n            SEEKING FOR I...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>"Victor Aloma"</td>
      <td>victorloma@netscape.net</td>
      <td></td>
      <td>ntcir-outgoing@nii.ac.jp</td>
      <td>16 Nov 2002</td>
      <td>Urgent Assistance</td>
      <td>The Director,\n\n\n\n            SEEKING FOR I...</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
