import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import datetime
import warnings

warnings.filterwarnings("ignore")
features = pd.read_csv('d:\\Temp\\temps.csv')
features.head()
features.shape

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(features['year'], features['month'], features['day'])]
len(dates)

dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
print(dates)

plt.style.use('fivethrityeight')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)

ax1.plot(dates, features['actual'])
ax1.set_xlabel('Date')
ax1.set_ylabel('Temp')
ax1.set_title('True Temp')

ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Temp')
ax2.set_title('Yesterday Temp')

ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date')
ax3.set_ylabel('Temp')
ax3.set_title('day before Yesterday Temp')

ax4.plot(dates, features['random'])
ax4.set_xlabel('Date')
ax4.set_ylabel('Temp')
ax4.set_title('Random Temp')

plt.tight_layout(pad=2)
fig.show()

features = pd.get_dummies(features)
print(features)

labels = np.array(features['actual'])
features = features.drop('actual', axis=1)
features = features.drop('random', axis=1)

feature_list = list(features.columns)
feature_list

features = np.array(features)
print(features.shape)

from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)
print(input_features[0])

model = tf.keras.Sequential()
"""
model.add(layers.Dense(16))
model.add(layers.Dense(32))
model.add(layers.Dense(1))

model.add(layers.Dense(16, kernel_initializer='random_normal'))
model.add(layers.Dense(32, kernel_initializer='random_normal'))
model.add(layers.Dense(1, kernel_initializer='random_normal'))
""""
model.add(layers.Dense(16, kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(32, kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1, kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))

model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.15, epochs=30, batch_size=64)
predict = model.summary()

print(input_features.shape)
predict = model.predict(input_features)
print(predict)

true_data = pd.DataFrame(data= {'date':dates, 'actual':labels})
predict_data = pd.DataFrame(data = {'date':dates, 'prediction':predict.reshape(-1)})

plt.plot(true_data['date'],true_data['actual'],'b-', label='Actual')
plt.plot(predict_data['date'],predict_data['prediction'],'ro', label='Prediction')
plt.legend()

plt.xlabel('Date')
plt.ylabel('Max Temp(F)')
plt.title('Actual and Predicted Temp(F)')
plt.show()
