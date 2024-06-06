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
butterfly_images, butterfly_labels = load_images("src/images/borboleta", 2)
fly_images, fly_labels = load_images("src/images/mosca", 3)
beetle_images, beetle_labels = load_images("src/images/tesourinha", 4)
cockroach_images, cockroach_labels = load_images("src/images/barata", 5)



images = np.array(spider_images + ant_images + butterfly_images + fly_images + beetle_images + cockroach_images)
labels = np.array(spider_labels + ant_labels + butterfly_labels + fly_labels + beetle_labels + cockroach_labels)

indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

split = int(0.8 * len(images))
X_train, X_test = images[:split], images[split:]
y_train, y_test = labels[:split], labels[split:]

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150, batch_size=50)
model.save('v2_1.keras')

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
