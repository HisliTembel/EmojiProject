import numpy as np
import cv2

import tensorflow_model_optimization as tfmot

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = np.load('3D.npy')
lab = np.load('Labels.npy')
data_norm = normalize(data)
print(data)
print(lab)

pivot3D = pd.DataFrame(data_norm, columns=["x1", "x2", "x3"])
pivot3D["label"] = lab
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
colors = ["red", "green", "blue", "orange", "black", "purple", "yellow", "grey", "pink", "brown"]
fig3 = plt.figure()
ax3 = Axes3D(fig3)
for i in range(0, 7):
    tmp = pivot3D[pivot3D.label == i]
    ax3.scatter(tmp.x1, tmp.x2, tmp.x3, c=colors[i], label=str(emotion_dict[i]))
ax3.legend(loc="upper left")
plt.title('3D view')
plt.show()
