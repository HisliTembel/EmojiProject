import numpy as np
import cv2
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import pandas as pd

from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=360, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=360, horizontal_flip=True)
my_batch_size = 16

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=my_batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=True,
    seed=42)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=my_batch_size,
    color_mode="grayscale",
    shuffle=True,
    class_mode="categorical")

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(4098, activation='relu'))
emotion_model.add(Dropout(0.2))
emotion_model.add(Dense(2048, activation='relu'))
emotion_model.add(Dropout(0.1))
emotion_model.add(Dense(512, activation='relu'))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.summary()
emotion_model.load_weights('emotion_model_checkpoint.hdf5')
emotion_model.pop()
emotion_model.pop()
emotion_model.pop()
emotion_model.pop()
emotion_model.pop()
emotion_model.pop()
emotion_model.summary()

train_filenames = train_generator.filenames
train_predict = emotion_model.predict_generator(train_generator, steps=np.ceil(28709 // my_batch_size))

print(train_predict.shape)


def SVD536(Mn, noise_amp_guess=False, N=1):
    U, S, VT = np.linalg.svd(Mn, full_matrices=False)  # SVD of python
    m, n = Mn.shape[0], Mn.shape[1]  # Taking size of the given matrix
    beta = n / m  # Cutoff parameter
    if not noise_amp_guess:  # Cutoff calculation started
        omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
        cutoff = omega * np.median(S)
        print(cutoff)
        print(S)
    else:
        lamda = np.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1) * np.sqrt(beta ** 2 + 14 * beta + 1)))
        cutoff = lamda * np.sqrt(n) * N  # Cutoff calculation ended
    r = np.max(np.where(S > cutoff))  # Deciding how many parameters will be used
    print(r + 1)
    Mres = U[:, :(r + 1)] @ np.diag(S[:(r + 1)]) @ VT[:(r + 1), :]  # Creating the low rank matrix.
    return Mres, r, S


def MatPrint(M, message='Matrix:'):
    """
    this fuction accepts a variable M, expected to be in the form of np.array
    and prints it more like a matrix
    Potential future work:
      digit size is currently fixed to 8, therefore it will not show arrays with long numbers properly
    """
    try:
        if M.ndim != 2:  # this function is meant to print only 2D arrays, if not 2D just print it
            print(message)
            print(M)
        else:
            print(message)
            spc = ' '
            sep = '|'
            for r in M:  # go over each row, i.e. 1 and 2 :)
                res = sep + spc
                for d in r:  # go over each element in the current row
                    res = res + '{:8.2f}'.format(d) + spc
                res = res + sep
                print(res)
    except:
        print(f'Is {M} really a meaningful numpy array?')


D, r, S = SVD536(train_predict)
MatPrint(S.T)
print(np.sum(np.multiply(S, S)))
print(S[0] * S[0])
print(S[0] * S[0] / np.sum(np.multiply(S, S)) * 100)
print(np.sum(S[0:430] * S[0:430]))
print(np.sum(S[0:430] * S[0:430]) / np.sum(np.multiply(S, S)) * 100)
print(np.sum(np.where(S > 10 ** -8, 1, 0)))

plt.xlabel("Singular Values")
plt.ylabel("Magnitude")
plt.plot(S, 'r*')
plt.show()
