import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

fashion_mnist = tf.keras.datasets.fashion_mnist
(traing_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
traing_images.shape

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(traing_images[1], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

traing_images = traing_images/255.0
test_images = test_images/255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    #plt.grid(False)
    plt.imshow(traing_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i]])

plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    # tf.keras.layers.Softmax()
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(traing_images, training_labels, epochs=10)
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f'Test accuracy: {test_accuracy}')

predictions = model.predict(test_images)
print(predictions[0])
np.argmax(predictions[0])