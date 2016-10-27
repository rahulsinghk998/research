#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers import LSTM, Input
import numpy

from keras.layers.extra import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten
from keras.layers import TimeDistributed, Activation, MaxPooling2D
import keras

import matplotlib.pyplot as plot

from server import client_generator


def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X, Y, Speed , Gas , Gear, Brake = tup
    #Y     = Y[:, -1]
    Speed = Speed[:,-1]
    Gas   = Gas[:, -1]
    Gear  = Gear[:, -1]
    Brake = Brake[:,-1]
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    yield X, Y


def get_model(time_len=1):
  time, ch, row, col =2, 3, 160, 320  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(time, ch, row, col), output_shape=(time, ch, row, col)))

  model.add(TimeDistributed(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same")))
  model.add(ELU())
  model.add(TimeDistributed(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same")))
  model.add(ELU())
  model.add(TimeDistributed(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same")))
  model.add(TimeDistributed(Flatten()))
  model.add(TimeDistributed(Dense(256)))

  #,activation='relu')))
  #model.add(TimeDistributed(Dense(128,activation='relu')))

  model.add(LSTM(output_dim=1,unroll=True, return_sequences=True))
  
  #model.add(Flatten())
  #model.add(Dropout(.2))
  #model.add(ELU())
  #model.add(Dense(512))
  #model.add(Dropout(.5))
  #model.add(ELU())
  #model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model
'''  model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='valid'),input_shape=(2, 3 , 160 , 320)))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(Activation('relu'))
  model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
  model.add(Dropout(0.25))

  model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='valid')))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(Activation('relu'))
  model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
  model.add(Dropout(0.25))

  model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='valid')))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(Activation('relu'))
  model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
  model.add(Dropout(0.25))

  model.add(TimeDistributed(Flatten()))
  model.add(TimeDistributed(Dense(256,activation='relu')))
  model.add(TimeDistributed(Dense(128,activation='relu')))

  model.add(LSTM(output_dim=1,unroll=True, return_sequences=True))

  model.compile(optimizer="adam", loss="mse")

  return model
  '''


 


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  model = get_model()
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










































        
"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""



"""
import h5py
from dask_generator import datagen
import numpy as np


def camera(time_len=1):
     hdf5_camera=[]
     c5x=[]
     filters=[]
     angle=[]
     speed=[]
     lastidx=0
     validation_path = [
         './dataset/camera/2016-06-02--21-39-29.h5',
         './dataset/camera/2016-06-08--11-46-01.h5'
       ]
     log_valid=[x.replace('camera','log') for x in validation_path]
     for cword, tword in zip(validation_path, log_valid):
     	with h5py.File(tword,"r") as t5:
            c5=h5py.File(cword,"r")
            hdf5_camera.append(c5)
            x=c5["X"]
            c5x.append((lastidx, lastidx+x.shape[0], x))
            speed_value = t5["speed"][:]
            steering_angle = t5["steering_angle"][:]
            idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")
            angle.append(steering_angle[idxs])
            speed.append(speed_value[idxs])

            goods = np.abs(angle[-1]) <= 200
            filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
            lastidx += goods.shape[0]

            #print(idxs)
     		#print(goods.shape[0])
            #print('rahul')
     		#print(lastidx)
            #help(filters)
            #print(steering_angle)
     		#print("......................................................")
     		#print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0]))
     angle = np.concatenate(angle, axis=0)
     speed = np.concatenate(speed, axis=0)
     filters = np.concatenate(filters, axis=0).ravel()
     filter_set=set(filters)
     #print("......................................................")
     #print(filters)
     #for a,b,c in c5x:
     #   print(a)
     #   print("......................................................")
     #   print(b)
     #   print("......................................................")
     #   print(c)
     #print(angle[0:10])
     print("......................................................")
     #angle_batch = np.copy(angle[0:10])[0:5,5:9]  
     #print(angle_batch)
     print("......................................................")
     #print ("training on %d/%d examples" % (filters.shape[0], angle.shape[0]))
     #print(c5x[1][2][18176][1][1])


if __name__ == '__main__':
	camera()
	
"""





















	#zi=[1,2,3]
	#z=camera()
	#for i in zi:
	#for i in c


"""
		  for cword, tword in zip(camera_names, logs_names):
    try:
      with h5py.File(tword, "r") as t5:
        c5 = h5py.File(cword, "r")
        hdf5_camera.append(c5)
        x = c5["X"]
        c5x.append((lastidx, lastidx+x.shape[0], x))
"""