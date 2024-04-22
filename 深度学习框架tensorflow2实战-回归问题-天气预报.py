import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras
import warnings
warnings.filterwarnings("ignore")
# 处理时间数据
import datetime

features = pd.read_csv('d:\\Temp\\temps.csv')
print(features.head())#显示读取的文件的前面几行,默认是前五行
print('数据维度:', features.shape)

# 分别得到年，月，日
years = features['year']
months = features['month']
days = features['day']

# datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
#print(dates)
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
print(dates)

# 准备画图
# 指定默认风格
plt.style.use('fivethirtyeight')

# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)#设置x坐标的标志为四十五度

# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('True Temp')

# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Yesterday Temp')

# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Yesterday of Yesterday Temp')

# 一个随机预测的温度
ax4.plot(dates, features['random'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('A random Temp')

plt.tight_layout(pad=2)#自动紧凑布局
plt.show()

print(features.head().shape)
# 独热编码
features = pd.get_dummies(features) #one-hot编码，对不是数字的会进行one-hot编码
print(features.head(1).shape)

# 标签，真实的温度
labels = np.array(features['actual'])

# 在特征中去掉标签
features= features.drop('actual', axis = 1)
features= features.drop('random', axis = 1)
#print(features.columns)
# 保存标头，即每列数据对应的含义保存一下
feature_list = list(features.columns)#

# 转换成合适的格式，只保留数据了
features = np.array(features)
print(features)
print(features.shape)

#preprocessing 为预处理库，里面封装了很多预处理操作
#无量纲化：
#标准化：（x-列均值）/ 列标准差
from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)
#print(input_features)
#print(input_features.shape)
#数据预处理完，接下来就是Tensorflow2的新展示了，使用keras的API
#模型堆叠
model = tf.keras.Sequential()
model.add(layers.Dense(16))     #全连接层，16个神经元
model.add(layers.Dense(32))     #全连接层，32个神经元
model.add(layers.Dense(1))      #全连接层，输出层
"""
#更改初始化方法
model.add(layers.Dense(16,kernel_initializer='random_normal'))
model.add(layers.Dense(32,kernel_initializer='random_normal'))
model.add(layers.Dense(1,kernel_initializer='random_normal'))

#加入正则化惩罚项
model.add(layers.Dense(16,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(32,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
"""
#compile相当于对网络进行配置，指定好优化器和损失函数等
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),
             loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.15, epochs=30, batch_size=64)#validation_split指的是验证集比例

model.summary()#展示网络层参数

predict = model.predict(input_features) #把输入传进去，得到我们的预测结果
print(predict.shape)

# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

# 同理，再创建一个来存日期和其对应的模型预测值
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predict.reshape(-1)})

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60')
plt.legend()

# 图名
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values')
plt.show()