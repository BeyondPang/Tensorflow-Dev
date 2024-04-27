import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

(x_train, y_train),(x_test, y_test) = mnist.load_data()

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

class CustomEealyStopping(tf.keras.callbacks.Callback):
    def __init__(self, min_accuracy=0.99):
        super(CustomEealyStopping, self).__init__
        self.min_accuracy = min_accuracy
    
    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')
        if current_accuracy is not None and current_accuracy >= self.min_accuracy:
            print('Reached 99% accuracy so cancelling training!')
            self.model.stop_training = True
            
early_stopping = CustomEealyStopping()

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(np.expand_dims(x_train, -1), y_train, epochs=10, validation_data=(np.expand_dims(x_test, -1), y_test), callbacks=[early_stopping])
test_acc, test_loss = model.evaluate(np.expand_dims(x_test, -1), y_test)
print(f"Test accuracy: {test_acc}")

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation loss')
plt.legend()
