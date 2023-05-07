import tensorflow as tf
import cv2
import os
import random
import pandas as pd

# Define the path to the folder containing the images
data_folder = 'images/preprocessed'

# Define the image size and channels
img_size = 300
channels = 3

# Define the batch size and number of epochs
batch_size = 32
epochs = 10

# Define the number of classes
num_classes = len(os.listdir(data_folder))

# Define the input shape
input_shape = (img_size, img_size, channels)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Split the data into training and validation sets
data = []
for class_folder in os.listdir(data_folder):
    class_path = os.path.join(data_folder, class_folder)
    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)
        data.append((image_path, class_folder))

random.shuffle(data)

split_index = int(len(data) * 0.8)
train_data = data[:split_index]
val_data = data[split_index:]

# Create training and validation data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1)

train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame(train_data, columns=['image_path', 'class']),
    x_col='image_path',
    y_col='class',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical')

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

val_generator = val_datagen.flow_from_dataframe(
    pd.DataFrame(val_data, columns=['image_path', 'class']),
    x_col='image_path',
    y_col='class',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical')

# Train the model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator)
