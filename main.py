import tensorflow as tf
import os
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
import cv2

categoris = ["door", "empty", "window"]

path = r'C:\Users\kwak8\PycharmProjects\Temp1\DATA_train\\'
path_test = r'C:\Users\kwak8\PycharmProjects\Temp1\DATA_test\\'
path_valid = r'C:\Users\kwak8\PycharmProjects\Temp1\DATA_valid\\'

# 모델 파일 저장시킬 위치
savepoint = r'C:\Users\kwak8\PycharmProjects\Version 1.0'

img_w = 60
img_h = 60
X = []
Y = []
X_V = []
Y_V = []
T = []
for index, cate in enumerate(categoris):
    # label = [0 for i in range(3)]
    # label[index] = 1
    image_dir = path + cate + '\\'
    valid_dir = path + cate + '\\'

    for top, dir, f in os.walk(image_dir):

        for filename in f:
            print(image_dir + filename)
            img = cv2.imread(image_dir + filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, None, fx=img_w/img.shape[1], fy=img_h/img.shape[0])

            '''
            cv2.imshow('ori', img)
            cv2.waitKey()
            '''
            X.append(img / 256)
            Y.append(index)

    for top, dir, f in os.walk(valid_dir):

        for filename in f:
            img = cv2.imread(valid_dir + filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, None, fx=img_w/img.shape[1], fy=img_h/img.shape[0])

            '''
            cv2.imshow('ori', img)
            cv2.waitKey()
            '''
            X_V.append(img / 256)
            Y_V.append(index)

temp = [[x, y] for x, y in zip(X, Y)]
random.shuffle(temp)
X = [n[0] for n in temp]
Y = [n[1] for n in temp]

X = np.array(X)
X = X.reshape(X.shape[0], img_h, img_w, 1)
# Y = np.array(Y)
Y = tf.keras.utils.to_categorical(Y, num_classes=3)

X_V = np.array(X_V)
X_V = X_V.reshape(X_V.shape[0], img_h, img_w, 1)
# Y_V = np.array(Y_V)
Y_V = tf.keras.utils.to_categorical(Y_V, num_classes=3)

model = Sequential()
model.add(Convolution2D(20, (3, 3),  activation='relu',
                        input_shape=(img_w, img_h, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Convolution2D(30, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Convolution2D(30, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))

# model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(X, Y, batch_size=20, epochs=15, verbose=2, validation_data=(X_V, Y_V))

model.save('RESULT_CALSSIFIRE.h5')
model.save(r'C:\Users\kwak8\PycharmProjects\Version 1.0\RESULT_CALSSIFIRE.h5')
model.summary()

name = []

for top, dir, f in os.walk(path_test):

    for filename in f:
        img = cv2.imread(path_test + filename, cv2.IMREAD_GRAYSCALE)

        name.append(filename)
        img = cv2.resize(img, None, fx=img_w/img.shape[1], fy=img_h/img.shape[0])

        T.append(img / 256)

T = np.array(T)
T = T.reshape(T.shape[0], img_h, img_w, 1)
pre = model.predict_classes(T)
for i in range(len(T)):
    print(name[i] + " --> Predict : " + str(categoris[pre[i]]))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/