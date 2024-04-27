import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

fashion_mnist = tf.keras.datasets.fashion_mnist
(traing_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
traing_images = traing_images/255.0
test_images = test_images/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

from sklearn.model_selection import train_test_split
traing_images, val_images, training_labels, val_labels = train_test_split(traing_images, training_labels, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

from keras.callbacks import EarlyStopping
# 创建 EarlyStopping 回调，监控验证损失，并设置耐心为 3
callback = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    np.expand_dims(traing_images, -1), 
    training_labels, 
    epochs=10, 
    validation_data=(np.expand_dims(val_images, -1), val_labels),
    callbacks=[callback])
)

test_loss, test_accuracy = model.evaluate(np.expand_dims(test_images, -1), test_labels, verbose=1)
print(f'Test accuracy: {test_accuracy}')

predictions = model.predict(test_images)
print(predictions[0])
np.argmax(predictions[0])