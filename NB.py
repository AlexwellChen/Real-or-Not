import pandas as pd
import numpy as np
import os
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

tweet = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')



# Remove URL
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# Remove HTML tags
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Remove emoji
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Remove punctuation
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

tweet['text']=tweet['text'].apply(lambda x : remove_punct(x))
tweet['text']=tweet['text'].apply(lambda x: remove_emoji(x))
tweet['text']=tweet['text'].apply(lambda x : remove_html(x))
tweet['text']=tweet['text'].apply(lambda x : remove_URL(x))

test['text']=test['text'].apply(lambda x : remove_punct(x))
test['text']=test['text'].apply(lambda x: remove_emoji(x))
test['text']=test['text'].apply(lambda x : remove_html(x))
test['text']=test['text'].apply(lambda x : remove_URL(x))

#准备训练数据
X_train = tweet["text"].tolist()
y_train = tweet["target"].tolist()
X_test = test["text"].tolist()


# 采用TfidfVectorizer提取文本特征向量
# 去除停用词
tfid_stop_vec = TfidfVectorizer(analyzer='word', stop_words='english')
x_tfid_stop_train = tfid_stop_vec.fit_transform(X_train)
x_tfid_stop_test = tfid_stop_vec.transform(X_test)

# 使用朴素贝叶斯分类器  分别对两种提取出来的特征值进行学习和预测

mnb_tfid_stop = MultinomialNB()
mnb_tfid_stop.fit(x_tfid_stop_train, y_train)   # 学习
mnb_tfid_stop_y_predict = mnb_tfid_stop.predict(x_tfid_stop_test)    # 预测

#提交数据
submission = pd.read_csv('data/sample_submission.csv')
submission['target'] = mnb_tfid_stop_y_predict
submission.to_csv('submission_2.csv',index=False)

