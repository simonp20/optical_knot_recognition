#conda activate DL

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

# Set the dimensions of the images
img_height = 300
img_width = 300

# Set the paths to the image folders
folder1_path = 'images/preprocessed/none'
folder2_path = 'images/preprocessed/overhand'
folder3_path = 'images/preprocessed/square'
folder4_path = 'images/preprocessed/bowline'
folder5_path = 'images/preprocessed/butterfly'
folder6_path = 'images/preprocessed/figure_eight'
folder7_path = 'images/preprocessed/figure_eight_on_a_bight'
folder8_path = 'images/preprocessed/sheet_bend'

# Load the images from each folder and create labels
folder1_images = []
folder1_labels = []
for file in os.listdir(folder1_path):
    img = cv2.imread(os.path.join(folder1_path, file))
    img_resized = cv2.resize(img, (img_height, img_width))
    folder1_images.append(img_resized)
    folder1_labels.append(0)

folder2_images = []
folder2_labels = []
for file in os.listdir(folder2_path):
    img = cv2.imread(os.path.join(folder2_path, file))
    img_resized = cv2.resize(img, (img_height, img_width))
    folder2_images.append(img_resized)
    folder2_labels.append(1)

folder3_images = []
folder3_labels = []
for file in os.listdir(folder3_path):
    img = cv2.imread(os.path.join(folder3_path, file))
    img_resized = cv2.resize(img, (img_height, img_width))
    folder3_images.append(img_resized)
    folder3_labels.append(2)
    
folder4_images = []
folder4_labels = []
for file in os.listdir(folder4_path):
    img = cv2.imread(os.path.join(folder4_path, file))
    img_resized = cv2.resize(img, (img_height, img_width))
    folder4_images.append(img_resized)
    folder4_labels.append(2)

print(len(folder4_images))
print(len(folder1_images))
print(len(folder1_images[0]))
print(len(folder4_images[0]))

# Combine the images and labels
images = np.concatenate((folder1_images, folder2_images, folder3_images, folder4_images), axis=0)
labels = np.concatenate((folder1_labels, folder2_labels, folder3_labels, folder4_images), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Normalize the pixel values of the images
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#define early stopping parameters for the model training
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=0.05)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))#, callbacks=[callback])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

model.save('knots_model')
