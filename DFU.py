

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
import PIL.Image as Image
import tensorflow as tf
from tensorflow import keras

import pathlib
data_dir = pathlib.Path('/content/drive/MyDrive/dl/Patches (1)')

data_dir

list(data_dir.glob("*/*.jpg"))[:5]
image_count=len(list(data_dir.glob("*/*.jpg")))
image_count

dfu_dict={
    'Abnormal(Ulcer)':list(data_dir.glob('Abnormal(Ulcer)/*')),
    'Normal(Healthy skin)':list(data_dir.glob('Normal(Healthy skin)/*')),
}

label_dict={
    'Abnormal(Ulcer)':0,
    'Normal(Healthy skin)':1,
}

from keras.preprocessing.image import ImageDataGenerator as IDG
train_gen = IDG(rescale=1./255, horizontal_flip=True, rotation_range=20, validation_split=0.2)

# Load Data
train_ds = train_gen.flow_from_directory(data_dir, target_size=(256,256), class_mode="sparse", subset='training', shuffle=True, batch_size=32)
valid_ds = train_gen.flow_from_directory(data_dir, target_size=(256,256), class_mode="sparse", subset='validation', shuffle=True, batch_size=32)

model=keras.Sequential()
model.add(keras.layers.Conv2D(filters=250,kernel_size=3,activation='relu',input_shape=[256,256,3]))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
model.add(keras.layers.Conv2D(filters=300,kernel_size=3,activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
model.add(keras.layers.Conv2D(filters=350,kernel_size=3,activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1000,activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1000,activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(800,activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(800,activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(11,activation="softmax"))

model.summary()

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics="accuracy")
history=model.fit(train_ds,validation_data=valid_ds,epochs=15)

model.evaluate(valid_ds)

import pandas as pd
pd.DataFrame(history.history).plot()

size=(256,256)
dfu_img =Image.open('/content/drive/MyDrive/dl/Patches (1)/Abnormal(Ulcer)/103.jpg').resize(size)

import matplotlib.pyplot as plt
plt.imshow(dfu_img)

dfu = np.array(dfu_img)/64.
dfu.shape

result=model.predict(dfu[np.newaxis,...])

if result[0][0] == 1:
    prediction ='Abnormal(Ulcer)'
else:
    prediction ='Normal(Healthy skin)'
print(prediction)

model.save("model.h5")