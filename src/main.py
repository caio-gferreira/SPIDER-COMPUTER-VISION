import tensorflow as tf
from tensorflow import keras
from services.open_cv_proccessor import OpenCVProcessor

image_processor = OpenCVProcessor()

training_data_dir = 'src/images/treinamento'
test_data_dir = 'src/images/teste'

training_images, training_labels = image_processor.get_proccessed_images(training_data_dir)
testing_images, testing_labels = image_processor.get_proccessed_images(test_data_dir)

train_datasets = tf.data.Dataset.from_tensor_slices((training_images, training_labels)).shuffle(10000).batch(32)
test_datasets = tf.data.Dataset.from_tensor_slices((testing_images, testing_labels)).batch(32)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

predictions = model(training_images[:1]).numpy()

tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(training_labels[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)

model.evaluate(testing_images,  testing_labels, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(testing_images[:5])
