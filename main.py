from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, MaxPooling2D
rows , col = 200, 200
count = 801
path1 = r"D:\Slosh AI\Pizza Classification\train\pizza"
path2 = r"D:\Slosh AI\Pizza Classification\train\Resized"
listing = os.listdir(path2)
num_samples = size(listing)
print(num_samples)
# for file in listing:
#     im = Image.open(path1 + '\\' + file)
#     img = im.resize((rows, col))
#     gray = img.convert('L')
#     gray.save(path2 + '\\' +str(count)+".JPEG", "JPEG")
#     count+=1
imlist = os.listdir(path2)
im1 = array(Image.open('D:\\Slosh AI\\Pizza Classification\\train\Resized' + '\\' + imlist[0]))
m, n = im1.shape[0:2]
imnbr = len(imlist)
immatrix = array([array(Image.open('D:\\Slosh AI\\Pizza Classification\\train\\Resized' + '\\' + im2)).flatten()for im2 in imlist], 'f')
label = np.ones((num_samples,), dtype=int)
label[0:801] = 0
label[801:] = 1
data, Label = shuffle(immatrix, label, random_state=2)
train_data = [data, Label]
img = immatrix[1599].reshape(rows, col)
nb_classes = 2
nb_filters = 32
nb_pool = 2
nb_conv = 2
(X, y) = (train_data[0], train_data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
print(X_train.shape)
print("test...")
X_train = X_train.reshape(X_train.shape[0],  rows, col,1)
X_test = X_test.reshape(X_test.shape[0],rows, col,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
i = 1439
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i, :])
model = Sequential()
inputShape = (rows, col,1)
model.add(Conv2D(nb_filters, (nb_conv, nb_conv),  padding="valid",input_shape=inputShape))
convout1 = Activation('relu')
model.add(convout1)
model.add(Conv2D(nb_filters,( nb_conv, nb_conv)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(X_train.shape)
print(X_test[0].shape)
model.fit(X_train, y_train, epochs=30)
model.save("D:\Slosh AI\Pizza Classification")
model.evaluate(X_train,y_train,batch_size=64)
print(Y_test)
predictions = model.predict(X_test)
print("predictions:", predictions)