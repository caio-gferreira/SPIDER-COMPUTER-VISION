import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def load_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.resize(img, (150, 150))
            images.append(img)
            labels.append(label)
    return images, labels

spider_images, spider_labels = load_images("src/images/aranha", 0)
ant_images, ant_labels = load_images("src/images/formiga", 1)


images = np.array(spider_images + ant_images)
labels = np.array(spider_labels + ant_labels)

indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

split = int(0.8 * len(images))
X_train, X_test = images[:split], images[split:]
y_train, y_test = labels[:split], labels[split:]

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
model.save('v1.keras')

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
