import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
from services.open_cv_proccessor import OpenCVProcessor

matplotlib.rcParams['figure.dpi'] = 150

image_processor = OpenCVProcessor()

training_data_dir = 'src/images/treinamento'
test_data_dir = 'src/images/teste'

training_images, training_labels = image_processor.get_proccessed_images(training_data_dir)
testing_images, testing_labels = image_processor.get_proccessed_images(test_data_dir)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

predictions = model(training_images)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(training_labels, predictions)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

history = model.fit(training_images, training_labels, 
                    epochs=10,
                    verbose=1,
                    validation_data=(testing_images, testing_labels),
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3
                    )])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()

# Plot custo de treino e validação
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Custo do Modelo')
plt.ylabel('Custo')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()

model.evaluate(testing_images,  testing_labels, verbose=1)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(testing_images)
