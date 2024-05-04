import os, shutil
import matplotlib.pylab as plt
import matplotlib.image as mpimg

base_dir = 'd:\\Temp\\cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cat_dir = os.path.join(train_dir, 'cats')
train_dog_dir = os.path.join(train_dir, 'dogs')

validation_cat_dir = os.path.join(validation_dir, 'cats')
validation_dog_dir = os.path.join(validation_dir, 'dogs')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 100
IMG_SHAPE = 64

def plotImages(images_arr):
    fig, axes = plt.subplots(1, len(images_arr), figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

img_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
#train_dir = 'D:\\Temp\\Dogs_n_Cats\\Augment'
train_datagen = img_gen.flow_from_directory(batch_size=BATCH_SIZE, directory=train_dir, shuffle=True, target_size=(IMG_SHAPE, IMG_SHAPE))

augmented_images = [train_datagen[0][0][0] for i in range(4)]
#plotImages(augmented_images)

#images, labels = train_datagen.next()
#plotImages(images[:4])

'''
train_iterator = iter(train_datagen)
augmented_images = []
for _ in range(4):
    images, labels = next(train_iterator)
    augmented_images.append(images[0])
    
plotImages(augmented_images)
'''

from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (3, 3)),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(128, (3, 3)),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

#model.summary()
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size=20,
    class_mode='binary')

val_gen = val_datagen.flow_from_directory(
    validation_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size=20,
    class_mode='binary')

history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=20,
    validation_data=val_gen,
    validation_steps=50,
    callbacks=[early_stopping],
    verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Train acc')
plt.plot(epochs, val_acc, label='Val acc')
plt.legend()
plt.title('Train and Validation Accuracy')
plt.figure()

plt.plot(epochs, loss, label='Train loss')
plt.plot(epochs, val_loss, label='Val loss')
plt.legend()
plt.title('Train and Validation Loss')
plt.figure()
plt.show()