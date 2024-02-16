import tensorflow as tf

from tensorflow.keras import layers, models

from helper.image_processor import ImageProcessor

image_processor = ImageProcessor()

training_data_dir = 'src/images/treinamento'
test_data_dir = 'src/images/teste'

training_images, training_labels = image_processor.load_images(training_data_dir)
testing_images, testing_labels = image_processor.load_images(test_data_dir)

training_images = training_images[..., tf.newaxis].astype("float32")
testing_images = testing_images[..., tf.newaxis].astype("float32")

train_datasets = tf.data.Dataset.from_tensor_slices((training_images, training_labels)).shuffle(10000).batch(32)
test_datasets = tf.data.Dataset.from_tensor_slices((testing_images, testing_labels)).batch(32)

print("\nROTULOS:", training_labels)

# Definir modelo CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_datasets, epochs=5)

test_loss, test_acc = model.evaluate(test_datasets)
print('\nAccuracy:', test_acc)
