import os
import csv
from tensorflow import keras
from PIL import Image
from numpy import *
model = keras.models.load_model("D:\Slosh AI\Pizza Classification")
print("Model Loaded")
rows, col = 200, 200
count = 1
path1 = r"D:\Slosh AI\Pizza Classification\test\images"
path2=r"D:\Slosh AI\Pizza Classification\test\Total Images"
listing = os.listdir(path2)
for file in listing:
    im = Image.open(path2 + '\\' + file)
    immatrix = array([array(im).flatten()], 'f')
    img11 = immatrix.reshape(rows, col)
    X_test = immatrix.reshape(immatrix.shape[0], rows,col, 1)
    X_test = X_test.astype('float32')
    X_test /= 255
    prediction = model.predict(X_test)
    print("Prediction: ",prediction)
    a =""
    if prediction[0][0] > prediction[0][1]:
        print("Not a Pizza")
        a = "Not a Pizza"
    else:
        print("Pizza")
        a = "Pizza"
    data = [file, a]
    with open('Test Results.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(data)
    file.close()