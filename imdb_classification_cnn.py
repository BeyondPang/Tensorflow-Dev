import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_features = 3000
sequence_length = 300
embedding_dim = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_features)

x_train = pad_sequences(x_train, maxlen=sequence_length)
x_test = pad_sequences(x_test, maxlen=sequence_length)

filter_size = [3, 4, 5]
def convolution():
    inn = layers.Input(shape=(sequence_length,embedding_dim,1))
    cnns = []
    for size in filter_size:
      conv = layers.Conv2D(filters=64, kernel_size=(size, embedding_dim), strides=1, padding='valid', activation='relu')(inn)
      pool = layers.MaxPool2D(pool_size=(sequence_length-size+1, 1), padding='valid')(conv)
      cnns.append(pool)
    outt = layers.concatenate(cnns)
    model = keras.Model(inputs=inn, outputs=outt)
    return model

def cnn_mulfilters():
  model = keras.Sequential([
      layers.Embedding(num_features, embedding_dim, input_length=sequence_length),
      layers.Reshape((sequence_length, embedding_dim, 1)),
      convolution(),
      layers.Flatten(),
      layers.Dense(10, activation='relu'),
      layers.Dropout(0.2),
      layers.Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

model = cnn_mulfilters()
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

loss, acc = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', acc)