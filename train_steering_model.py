#!/usr/bin/env python
"""
Steering angle prediction model

http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
##Model checkpoints in keras:: https://keras.io/callbacks/#modelcheckpoint
#Checkpoints:: http://machinelearningmastery.com/check-point-deep-learning-models-keras/
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
----------------------------------------------------------------------------------------------------------
#Keras Callback:: https://keras.io/callbacks/
#http://stackoverflow.com/questions/36895627/python-keras-creating-a-callback-with-one-prediction-for-each-epoch?noredirect=1&lq=1
class prediction_history(Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.predhis=(model.predict(predictor_train))

#Calling the subclass
predictions=prediction_history()

#Executing the model.fit of the neural network
model.fit(X=predictor_train, y=target_train, nb_epoch=2, batch_size=batch,validation_split=0.1,callbacks=[predictions]) 

#Printing the prediction history
print predictions.predhis
----------------------------------------------------------------------------------------------------------
Visualizing the intermediate layers
https://github.com/fchollet/keras/issues/41
https://github.com/fchollet/keras/issues/2890
1. import theano
get_activations = theano.function([model.layers[0].input], model.layers[1].output(train=False), allow_input_downcast=True)
activations = get_activations(X_batch) # same result as above
----------------------------------------------------------------------------------------------------------
issue with visualization
1. https://github.com/fchollet/keras/issues/3216
----------------------------------------------------------------------------------------------------------
Check model.pop() working
----------------------------------------------------------------------------------------------------------
#put trained cnn model weights to train the rnn model#Intermediate layer visualization
#https://keras.io/visualization/
"""
import os
import argparse
import json
import keras
import h5py
import numpy as np
import keras.callbacks as cb
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.models import Sequential
from keras.engine.topology import Merge
from keras.models import model_from_json
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import TimeDistributed, Activation, MaxPooling2D
from server import client_generator
from keras.layers.core import RepeatVector
import predict_model

#layer_indices=[0,1,2,6,7]
#https://groups.google.com/forum/#!topic/keras-users/UzKGcXtUucU
def load_custom_weights(model, filepath, layer_indices):
  f = h5py.File(filepath, mode='r')
  #g = f['graph']
  #weights = [g['param_{}'.format(p)] for p in layer_indices]
  model.set_weights(weights)
  f.close()



def pop_layer(model):
  if not model.outputs:
    raise Exception('Sequential model cannot be popped: model is empty.')
  model.layers.pop()
  if not model.layers:
    model.outputs = []
    model.inbound_nodes = []
    model.outbound_nodes = []
  else:
    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]
  model.built = False



def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    Image, Steer, Speed , Gas , Gear, Brake = tup

    if Image.shape[1] == 1:  # no temporal context
      Image = Image[:, -1]
      #Steer = Steer[:, -1]
      #Speed = Speed[:,-1]
      Gas   = Gas[:, -1]
      Gear  = Gear[:, -1]
      Brake = Brake[:,-1]
      yield [Image,Speed], Steer
    else:
      yield [Image,Speed], Steer  #Need to change according to the modelling parameter



class LossHistory(cb.Callback):
  def on_train_begin(self, logs={}):
      self.losses = []

  def on_batch_end(self, batch, logs={}):
      batch_loss = logs.get('loss')
      self.losses.append(batch_loss)



def plot_losses(losses):
  x=np.array(losses)
  np.savetxt("./outputs/steering_model/loss_plot.txt",losses, fmt="%5.2f")

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(losses)
  ax.set_title('Loss per batch')
  fig.savefig("./outputs/steering_model/loss_plot.png",format='eps', dpi=1000)
  fig.show()
  #fig.pause(0.005)



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
  history = LossHistory()

  model = predict_model.get_model(args.time)
  model.fit_generator(
    gen(20, args.host, port=args.port),
    samples_per_epoch=10000,
    nb_epoch=args.epoch,
    callbacks=[history], 
    #show_accuracy=True,
    validation_data=gen(20, args.host, port=args.val_port),
    nb_val_samples=1000,
    verbose=2
  )

  plot_losses(history.losses)
  #score = model.evaluate(X_test, y_test, batch_size=16, show_accuracy=True)
  print("Saving model weights and configuration file.")
  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")

  model.save_weights("./outputs/steering_model/steering_angle.keras", True)
  with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)