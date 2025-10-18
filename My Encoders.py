import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from random import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

encoder_2D = Sequential()
encoder_2D.add(Dense(1024, activation='relu', input_shape=(2048,)))
encoder_2D.add(Dense(256, activation='relu'))
encoder_2D.add(Dense(64, activation='relu'))
encoder_2D.add(Dense(16, activation='relu'))
encoder_2D.add(Dense(2, activation='relu'))
encoder_2D.add(Dense(16, activation='relu'))
encoder_2D.add(Dense(64, activation='relu'))
encoder_2D.add(Dense(256, activation='relu'))
encoder_2D.add(Dense(1024, activation='relu'))
encoder_2D.add(Dense(2048, activation='sigmoid'))
encoder_2D.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

encoder_3D = Sequential()
encoder_3D.add(Dense(1024, activation='relu', input_shape=(2048,)))
encoder_3D.add(Dense(256, activation='relu'))
encoder_3D.add(Dense(64, activation='relu'))
encoder_3D.add(Dense(16, activation='relu'))
encoder_3D.add(Dense(3, activation='relu'))
encoder_3D.add(Dense(16, activation='relu'))
encoder_3D.add(Dense(64, activation='relu'))
encoder_3D.add(Dense(256, activation='relu'))
encoder_3D.add(Dense(1024, activation='relu'))
encoder_3D.add(Dense(2048, activation='sigmoid'))
encoder_3D.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

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
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.summary()
emotion_model.load_weights('emotion_model_checkpoint.hdf5')
emotion_model.pop()
emotion_model.pop()
emotion_model.pop()
emotion_model.summary()

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    seed=42)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical")

train_filenames = train_generator.filenames
train_nb_samples = len(train_filenames)
train_predict = emotion_model.predict_generator(train_generator, steps=np.ceil(train_nb_samples / 32))

validation_filenames = validation_generator.filenames
validation_nb_samples = len(validation_filenames)
validation_predict = emotion_model.predict_generator(validation_generator, steps=np.ceil(validation_nb_samples / 32))

earlyStop = EarlyStopping(monitor='val_loss', patience=50)
checkpoint_2D = ModelCheckpoint(filepath='encoder_2d_checkpoint.hdf5', monitor='val_loss',
                                save_best_only=True, mode='auto')
checkpoint_3D = ModelCheckpoint(filepath='encoder_3d_checkpoint.hdf5', monitor='val_loss',
                                save_best_only=True, mode='auto')

encoder_2D.fit(train_predict, train_predict, epochs=10, validation_data=(validation_predict, validation_predict),
               verbose=2, callbacks=[checkpoint_2D, earlyStop])
encoder_3D.fit(train_predict, train_predict, epochs=10, validation_data=(validation_predict, validation_predict),
               verbose=2, callbacks=[checkpoint_3D, earlyStop])

encoder_2D.summary()
encoder_3D.summary()


def calculatePerformance2D(x):
    predictions = encoder_2D.predict(x)
    actuals = x.copy()
    return mean_absolute_error(actuals, predictions), sqrt(mean_squared_error(actuals, predictions))


def calculatePerformance3D(x):
    predictions = encoder_3D.predict(x)
    actuals = x.copy()
    return mean_absolute_error(actuals, predictions), sqrt(mean_squared_error(actuals, predictions))


validation_perf_2D = calculatePerformance2D(validation_predict)
train_perf_2D = calculatePerformance2D(train_predict)

validation_perf_3D = calculatePerformance3D(validation_predict)
train_perf_3D = calculatePerformance3D(train_predict)

if (abs(validation_perf_2D[0] - train_perf_2D[0]) < 0.01) and (abs(validation_perf_2D[1] - train_perf_2D[1]) < 0.01):
    encoder_2D.save_weights("encoder_2D_itself.h5")
    print("2D weights are saved.")

if (abs(validation_perf_3D[0] - train_perf_3D[0]) < 0.01) and (abs(validation_perf_3D[1] - train_perf_3D[1]) < 0.01):
    encoder_3D.save_weights("encoder_3D_itself.h5")
    print("3D weights are saved.")
