import tensorflow as tf
from tensorflow import keras
from services.open_cv_proccessor import OpenCVProcessor
import os

image_processor = OpenCVProcessor()

training_data_dir = 'src/images/treinamento'
test_data_dir = 'src/images/teste'

train_images, train_labels = image_processor.get_proccessed_images(training_data_dir)
test_images, test_labels = image_processor.get_proccessed_images(test_data_dir)

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.sigmoid),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    return model

def save_checkpoint_model(model, checkpoint_path):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    model.fit(train_images, 
            train_labels,  
            epochs=10,
            validation_data=(test_images, test_labels),
            callbacks=[cp_callback])
    
    return model

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a `.keras` zip archive.
model.save('./src/model/v1.keras')

