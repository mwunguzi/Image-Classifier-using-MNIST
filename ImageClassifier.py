import tensorflow as tf
from tensorflow import keras

fashion_dataset = keras.datasets.fashion_mnist

# loading the fashion dataset into train and test (70,000 gray images, 28x28 )
(x_train,y_train), (x_test,y_test) = fashion_dataset.load_data()

print(x_train.shape)
print(x_train.dtype)
