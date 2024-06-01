# 导包
import re
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

cn_model = KeyedVectors.load_word2vec_format('vectors/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2', binary=False, unicode_errors='ignore')

## 4.读取训练数据
pos_file_list = os.listdir('data/pos')
neg_file_list = os.listdir('data/neg')
pos_file_list = [f'data/pos/{x}' for x in pos_file_list]
neg_file_list = [f'data/neg/{x}' for x in neg_file_list]
pos_neg_file_list = pos_file_list + neg_file_list
# 读取所有的文本，放入到x_train,前3000是正向样本，后3000负向样本
x_train = []
for file in pos_neg_file_list:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    x_train.append(text)

x_train = np.array(x_train)
y_train = np.array([1] * len(pos_file_list) + [0] * len(neg_file_list))
print(x_train.shape, y_train.shape)

np.random.seed(42)
np.random.shuffle(x_train)
np.random.seed(42)
np.random.shuffle(y_train)

x_train_tokens = []
for x in x_train:
  words = jieba.cut(x)
  word_list = [word for word in words]
  for i, word in enumerate(word_list):
    try:
      word_list[i] = cn_model.key_to_index[word]
    except:
      word_list[i] = 0
  x_train_tokens.append(word_list)
  
token_count = [len(x) for x in x_train_tokens]
length_frequency = {length: token_count.count(length) for length in set(token_count)}

# 画图查看词的长度分布
plt.bar(range(len(length_frequency)), list(length_frequency.values()), align='center')
plt.ylabel('Frequency')  # 更正 y 轴标签为频率
plt.xlabel('Tokens Length')  # 更正 x 轴标签为词的长度
#plt.xticks(range(len(length_frequency)), list(length_frequency.keys()))  # 设置 x 轴的刻度为不同的长度值

# 显示图表
plt.show()
    
tokens_length = int(np.mean(tokens_count) + 2 * np.std(tokens_count))
np.sum(np.array(tokens_count) < tokens_length) / len(tokens_count)

def reverse_tokens(tokens):
  #return ''.join(cn_model.index_to_key[token] for token in tokens)
  text = ''
  for index in tokens:
    if index == 0:
      break
    else:
      text += cn_model.index_to_key[index]
  return text

num_words = 50000
num_dims = 300

embedding_matrix = np.zeros((num_words, num_dims))
for i in range(num_words):
  embedding_matrix[i] = cn_model[cn_model.index_to_key[i]]
embedding_matrix = embedding_matrix.astype('float32')

x_train_tokens_pad = tf.keras.preprocessing.sequence.pad_sequences(x_train_tokens, maxlen=token_length, padding='pre', truncating='pre')
x_train_tokens_pad[x_train_tokens_pad>num_words] = 0

x_tokens_train, x_tokens_test, y_train, y_test = train_test_split(x_train_tokens_pad, y_train, test_size=0.1, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, num_dims, weights=[embedding_matrix], input_length=token_length, trainable=False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(x_tokens_train, y_train, epochs=10, batch_size=32, validation_data=(x_tokens_test, y_test))

loss, accuracy = model.evaluate(x_tokens_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

def predict_sentiment(text):
  cut = jieba.cut(text)
  cut_list = [word for word in cut]
  for i, word in enumerate(cut_list):
    try:
      cut_list[i] = cn_model.key_to_index[word]
    except KeyError:
      cut_list[i] = 0
  token_pad = tf.keras.preprocessing.sequence.pad_sequences([cut_list], maxlen=token_length, padding='pre', truncating='pre')
  token_pad[token_pad>num_words] = 0
  result = model.predict(token_pad)
  if result[0][0] > 0:
    return(f'正:{ 1 / (1 + np.exp(-result[0][0]))}')
  else:
    return(f'负:{1 -  1 / (1 + np.exp(-result[0][0]))}')
 

test_list = [
'酒店设施不是新的，服务态度很不好',
'酒店卫生条件非常不好',
'床铺非常舒适',
'房间很冷，还不给开暖气',
'房间很凉爽，空调冷气很足',
'酒店环境不好，住宿体验很不好',
'房间隔音不到位' ,
'晚上回来发现没有打扫卫生,心情不好',
'因为过节所以要我临时加钱，比团购的价格贵',
'房间很温馨，前台服务很好,'
]

for text in test_list:
  print(text)
  print(predict_sentiment(text))
