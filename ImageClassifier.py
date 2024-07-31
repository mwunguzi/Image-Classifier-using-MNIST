
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_dataset = keras.datasets.fashion_mnist

# loading the fashion dataset into train and test (70,000 gray images, 28x28 )
(x_train,y_train), (x_test,y_test) = fashion_dataset.load_data()


#data spliting nad preprocessing
# pixel values are between 0 and 255 to change it to a range of 0,1 we use normalization(MinMaxScaler)
# MinMaxScaler = (x_i - x_min)/(x_max - x_min) in our case X_min is 0 and x_max is 255
x_train = x_train[:5000]/255.0
x_val = x_train[5000:]/255.0
y_train = y_train[:5000]
y_val = y_train[5000:]

print("Train size = ", x_train.shape)

fashion_classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# MLP Creation and configuration
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(212,activation='relu'))
model.add(keras.layers.Dense(40,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

print(model.summary())

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=35,validation_data=(x_val,y_val))
print("\n")

print("Accuracy = ",model.evaluate(x_test,y_test))





