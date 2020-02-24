import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import re
import string
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing .sequence import  pad_sequences


# 读入数据
tweet = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 首先进行数据清洗
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

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train + X_test)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train)
X_test = pad_sequences(X_test,)

word_index = tokenizer.word_index
nb_words = len(word_index)  # 29363
print(nb_words)

all_data=pd.concat([tweet['text'],test['text']])
model = word2vec.Word2Vec([[word for word in sentence.split(' ')] for sentence in all_data.values],
                     size=200, window=5, iter=10, workers=11, seed=2018, min_count=2)

# 建立embedding矩阵
nb_words = min(max_features, len(word_index))
embedding_word2vec_matrix = np.zeros((nb_words, 200))  # (95000, 200)
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = model[word] if word in model else None
    if embedding_vector is not None:
        count += 1
        embedding_word2vec_matrix[i] = embedding_vector
    else:
        unk_vec = np.random.random(200) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        embedding_word2vec_matrix[i] = unk_vec



