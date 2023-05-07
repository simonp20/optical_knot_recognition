import tensorflow as tf
import cv2
import numpy as np
import os

# Define the path to the saved model
model_path = 'saved_model.pb'

# Define the paths to the folders containing the images
class_folders = ['class1', 'class2', 'class3']

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Define the image size and channels
img_size = 224
channels = 3

# Define the batch size
batch_size = 32

# Create a test generator for each class
test_generators = []
for folder in class_folders:
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        f'path/to/folders/{folder}',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')
    test_generators.append(test_generator)

# Evaluate the model on the test generators
for i, test_generator in enumerate(test_generators):
    print(f'Evaluating on class {i+1}...')
    loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

