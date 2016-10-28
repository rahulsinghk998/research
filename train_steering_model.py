#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
import keras
import numpy
from keras.layers import LSTM
from keras.models import Sequential
from keras.engine.topology import Merge
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import TimeDistributed, Activation, MaxPooling2D

import matplotlib.pyplot as plot

from server import client_generator


def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    Image, Steer, Speed , Gas , Gear, Brake = tup

    if Image.shape[1] == 1:  # no temporal context
      Image = Image[:, -1]
      Steer = Steer[:, -1]
      Speed = Speed[:,-1]
      Gas   = Gas[:, -1]
      Gear  = Gear[:, -1]
      Brake = Brake[:,-1]
      yield Image, Steer
    else:
      yield [Image,Speed], Steer  #Need to change according to the modelling parameter


def get_model(time_len=1):
  if time_len>1:
    time, ch, row, col = time_len, 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(time, ch, row, col), output_shape=(time, ch, row, col)))

    model.add(TimeDistributed(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same")))
    model.add(ELU())
    model.add(TimeDistributed(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same")))
    model.add(ELU())
    model.add(TimeDistributed(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same")))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(256)))

    model1=Sequential()
    model1.add(Lambda(lambda x:x, input_shape=(time_len,1), output_shape=(time_len,1)))
    merge=Sequential()
    merge.add(Merge([model, model1], mode='concat', concat_axis=2))
    #merge.add(Merge([model, tensor], mode='concat', concat_axis=2))
    merge.add(LSTM(output_dim=1, unroll=True, return_sequences=True))

    #,activation='relu')))
    #model.add(TimeDistributed(Dense(128,activation='relu')))
    #model.add(LSTM(output_dim=1,unroll=True, return_sequences=True))
    model.compile(optimizer="adam", loss="mse")
    model1.compile(optimizer="adam", loss="mse")
    merge.compile(optimizer="adam", loss="mse")

    return merge

  else:
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(ch, row, col),
              output_shape=(ch, row, col)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model
  #tensor = Input((time_len,1))




if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--time', type=int, default=1, help='Number of time stamps.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  model = get_model(args.time)
  model.fit_generator(
    gen(20, args.host, port=args.port),
    samples_per_epoch=10000,
    nb_epoch=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port),
    nb_val_samples=1000
  )
  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")

  model.save_weights("./outputs/steering_model/steering_angle.keras", True)
  with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
