import os
import argparse
import json
import keras
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.models import Sequential
from keras.engine.topology import Merge
from keras.models import model_from_json
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import TimeDistributed, Activation, MaxPooling2D
from server import client_generator
import cv2

with open("./outputs/steering_model/steering_angle.json", 'r') as jfile:
  model = model_from_json(json.load(jfile))

model.compile("sgd", "mse")
model.load_weights("./outputs/steering_model/steering_angle.keras")

time, ch, row, col = 1, 3, 160, 320
model1 = Sequential()
model1.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
model1.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", weights=model.layers[1].get_weights()))

dataset="2016-06-02--21-39-29"
log = h5py.File("dataset/log/"+dataset+".h5", "r")
cam = h5py.File("dataset/camera/"+dataset+".h5", "r")
img = np.array((cam['X'][log['cam1_ptr'][200000]]))
predicted_steers = model1.predict(img[None, :, :, :])

cv2.imwrite("./outputs/steering_model/layer_output/image.jpg", img[0])

for i in range(0,16):
  predicted_steers.shape
  y=predicted_steers[0][i]*255
  x=cv2.resize(y,(320,160))
  cv2.imwrite("./outputs/steering_model/layer_output/cnn_layer1_{}.jpg".format(i),x)

model1.add(ELU())
model1.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same" , weights=model.layers[3].get_weights()))
model1.compile("sgd", "mse")
predicted_steers = model1.predict(img[None, :, :, :])

for i in range(0,32):
  predicted_steers.shape
  y=predicted_steers[0][i]*255
  x=cv2.resize(y,(320,160))
  cv2.imwrite("./outputs/steering_model/layer_output/cnn_layer2_{}.jpg".format(i),x)

model1.add(ELU())
model1.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", weights=model.layers[5].get_weights()))
model1.compile("sgd", "mse")
predicted_steers = model1.predict(img[None, :, :, :])

for i in range(0,64):
  predicted_steers.shape
  y=predicted_steers[0][i]*255
  x=cv2.resize(y,(320,160))
  cv2.imwrite("./outputs/steering_model/layer_output/cnn_layer3_{}.jpg".format(i),x)

