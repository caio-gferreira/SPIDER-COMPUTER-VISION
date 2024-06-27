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

def load_image_test(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.resize(img, (150, 150))

            images.append(img)
            label = int(os.path.splitext(filename)[0])
            labels.append(label)
    return images, labels


fly_images, fly_labels = load_images("src/images/mosca", 0)
ant_images, ant_labels = load_images("src/images/formiga", 1)
spider_images, spider_labels = load_images("src/images/aranha", 2)
cockroach_images, cockroach_labels = load_images("src/images/barata", 3)
butterfly_images, butterfly_labels = load_images("src/images/borboleta", 4)

test_images, test_labels = load_image_test("src/images/test_images")

class_names = ['mosca', 'formiga', 'aranha', 'barata', 'borboleta']



train_images = np.array(spider_images + ant_images + butterfly_images + fly_images + cockroach_images)
train_labels = np.array(spider_labels + ant_labels + butterfly_labels + fly_labels + cockroach_labels)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

model.save('./src/model/v3_2.keras')