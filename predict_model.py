import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers import LSTM, Input
import numpy
from keras.layers import TimeDistributed, Activation, MaxPooling2D
from keras.engine.topology import Merge
import keras
from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plot
from keras.layers.core import RepeatVector


#Now actual time distributed version can be implemented# Currently have some doubt for RepeatVector layer
def get_model(time_len=1):

  time, ch, row, col = time_len, 3, 160, 320  # camera format

  model1 = Sequential()
  model1.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
  model1.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model1.add(ELU())
  model1.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model1.add(ELU())
  model1.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model1.add(Flatten())
  model1.add(Dropout(.2))
  model1.add(ELU())
  model1.add(Dense(512))
  model1.add(Dropout(.5))
  model1.add(ELU())
  model1.add(Dense(1))
  model1.load_weights("./outputs/steering_model_trained/steering_angle.keras")
  #model.compile(optimizer="adam", loss="mse")
  #rnn.add(TimeDistributed(vgg_model, input_shape=(10, 3, 224, 224)))
  #rnn.add(LSTM(10, activation='tanh'))
  #rnn.add(Dense(1, activation='sigmoid'))


  #Loading Trained Weights :: Model2
  model2 = Sequential()
  model2.add(Lambda(lambda x: x/127.5 - 1., input_shape=(time, ch, row, col), output_shape=(time, ch, row, col)))
  model2.add(TimeDistributed(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", weights=model1.layers[1].get_weights(), trainable=False)))
  model2.add(TimeDistributed(ELU()))
  model2.add(TimeDistributed(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", weights=model1.layers[3].get_weights(), trainable=False)))
  model2.add(TimeDistributed(ELU()))
  model2.add(TimeDistributed(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", weights=model1.layers[5].get_weights(), trainable=False)))
  model2.add(TimeDistributed(Flatten()))
  model2.add(TimeDistributed(Dropout(.2)))
  model2.add(TimeDistributed(ELU()))
  model2.add(TimeDistributed(Dense(512, weights=model1.layers[9].get_weights(), trainable=False)))
  #model2.add(TimeDistributed(RepeatVector(time_len)))


  #Merge Model:: Model3
  model3=Sequential()
  #model3.add(Dense(1, input_dim=(time_len,1))) --> need to check this layer
  model3.add(Lambda(lambda x:x, input_shape=(time_len,1), output_shape=(time_len,1)))
  merge=Sequential()
  merge.add(Merge([model2, model3], mode='concat', concat_axis=2))
  merge.add(LSTM(output_dim=1, unroll=True, return_sequences=True))


  merge.compile(optimizer="adam", loss="mse")
  return merge


if __name__ == '__main__':
  print("Model Compiled.....")
  get_model().summary()










"""
def get_model(time_len=1):
  if time_len>1:
    time, ch, row, col = time_len, 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(time, ch, row, col), output_shape=(time, ch, row, col)))

    model.add(TimeDistributed(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", trainable=False)))
    model.add(TimeDistributed(ELU()))
    model.add(TimeDistributed(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", trainable=False)))
    model.add(TimeDistributed(ELU()))
    model.add(TimeDistributed(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", trainable=False)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(ELU()))
    model.add(TimeDistributed(Dense(512,trainable=False)))
    model.add(TimeDistributed(Dropout(.5)))
    model.add(TimeDistributed(ELU()))
    model.add(TimeDistributed(Dense(1)))

    '''Loading weights of trained CNN Model'''
    model.load_weights("./outputs/steering_model_trained/steering_angle.keras")
    model.layers.pop()
    model.layers.pop()
    model.add(TimeDistributed(Dense(256)))
    model.compile(optimizer="adam", loss="mse")

    model1=Sequential()
    model1.add(Lambda(lambda x:x, input_shape=(time_len,1), output_shape=(time_len,1)))
    merge=Sequential()
    merge.add(Merge([model, model1], mode='concat', concat_axis=2))
    print(merge.output_shape)
    merge.add(LSTM(output_dim=1, unroll=True, return_sequences=True))
    #merge.add(LSTM(10, activation='tanh'))
    #merge.add(TimeDistributed(Dense(1, activation='sigmoid')))

    #,activation='relu')))
    #model.add(TimeDistributed(Dense(128,activation='relu')))

    model1.compile(optimizer="adam", loss="mse")
    merge.compile(optimizer="adam", loss="mse")

    return merge


hidden_neurons = 300
model = Sequential()
#model.add(Masking(mask_value=0, input_shape=(input_dim,)))
model.add(LSTM(hidden_neurons, return_sequences=False, batch_input_shape=X_train.shape))
# only specify the output dimension
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(X_train, y_train, nb_epoch=10, validation_split=0.05)

# calculate test set MSE
preds = model.predict(X_test).reshape(len(y_test))
MSE = np.mean((preds-y_test)**2)

  a=Input((10,))
  d=Dense(10)(a)
  print(d.output_shape)
  #a=Input((2,1))
  #d=Dense((2,1))(a)

  #print(d.output_shape)
  #print(model.output_shape)

  mode = Model(input=[model, a], output=merge([model, d]))
  """