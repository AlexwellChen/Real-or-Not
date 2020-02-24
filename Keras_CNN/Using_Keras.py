from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
import re
import string
import numpy as np
from keras import *
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

def clean_data(df):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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
        
    df['text']=df['text'].apply(lambda x : remove_punct(x))
    df['text']=df['text'].apply(lambda x: remove_emoji(x))
    df['text']=df['text'].apply(lambda x : remove_html(x))
    df['text']=df['text'].apply(lambda x : remove_URL(x))

#构建TextCNN模型
#模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
def TextCNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs):
    main_input = Input(shape=(50,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(2, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
    one_hot_labels = utils.to_categorical(y_train)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=10)
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(int, result_labels))
    submission = pd.read_csv('data/sample_submission.csv')
    submission['target'] = y_predict
    submission.to_csv('submission_2.csv',index=False)
    # print('准确率', sklearn.metrics.accuracy_score(y_test, y_predict))
    # print('平均f1-score:', sklearn.metrics.f1_score(y_test, y_predict, average='weighted'))
    print("预测完成")

if __name__=='__main__':
    dataset = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    clean_data(dataset)
    clean_data(test)
    tokenizer=Tokenizer()  #创建一个Tokenizer对象

    stopworddic = set({'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'})  
    #  去除停用词
    dataset['text'] = [i for i in dataset['text'] if i not in stopworddic ]
    test['text'] = [i for i in test['text'] if i not in stopworddic ]
    #fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    fit_text = dataset['text'].tolist()+test['text'].tolist()
    tokenizer.fit_on_texts(fit_text)
    vocab=tokenizer.word_index #得到每个词的编号
    # x_train, x_test, y_train, y_test = train_test_split(dataset['text'].tolist(), dataset['target'].tolist(), test_size=0.1)
    x_train = dataset['text'].tolist()
    x_test = test['text'].tolist()
    y_train = dataset['target'].tolist()
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    x_train_word_ids=tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    #序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    x_train_padded_seqs=pad_sequences(x_train_word_ids,maxlen=50) #将超过固定值的部分截掉，不足的在最前面用0填充
    x_test_padded_seqs=pad_sequences(x_test_word_ids, maxlen=50)

    # 使用TextCNN训练
    TextCNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs)
