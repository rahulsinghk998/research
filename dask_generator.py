"""
This file is named after `dask` for historical reasons. We first tried to
use dask to coordinate the hdf5 buckets but it was slow and we wrote our own
stuff.
"""
import numpy as np
import h5py
import time
import logging
import traceback

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def concatenate(camera_names, time_len):
  logs_names = [x.replace('camera', 'log') for x in camera_names]

  angle = []        # steering angle of the car
  speed = []        # speed of the car
  gas = []          # gas padel position of car
  gear_choice = []  # gear choice of car
  brake = []        # brake position of car

  hdf5_camera = []  # the camera hdf5 files need to continue open
  c5x = []          # stores the all set of input dataset
  filters = []      # stores all the 'good' time stamps i.e. when Steering_angle<=Threshold_steering
  lastidx = 0

  for cword, tword in zip(camera_names, logs_names):
    try:
      with h5py.File(tword, "r") as t5:
        c5 = h5py.File(cword, "r")
        hdf5_camera.append(c5)
        x = c5["X"]
        c5x.append((lastidx, lastidx+x.shape[0], x))

        speed_value = t5["speed"][:]
        steering_angle = t5["steering_angle"][:]
        gas_position = t5["gas"][:]
        gear_value = t5["gear_choice"][:]
        brake_position = t5["brake"][:]  #Choosen 1 but has 3 set of brake values namely 1. Brake  2.Brake_computer  3.Brake_user
        
        idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
        angle.append(steering_angle[idxs])
        speed.append(speed_value[idxs])
        gas.append(gas_position[idxs])
        gear_choice.append(gear_value[idxs])
        brake.append(brake_position[idxs])

        goods = np.abs(angle[-1]) <= 200

        filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
        lastidx += goods.shape[0]
        # check for mismatched length bug
        print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0]))
        if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
          raise Exception("bad shape")

    except IOError:
      import traceback
      traceback.print_exc()
      print ("failed to open", tword)

  angle         = np.concatenate(angle, axis=0)
  speed         = np.concatenate(speed, axis=0)
  gas           = np.concatenate(gas, axis=0)
  gear_choice   = np.concatenate(gear_choice, axis=0)
  brake         = np.concatenate(brake, axis=0)
  filters       = np.concatenate(filters, axis=0).ravel()
  print ("training on %d/%d examples" % (filters.shape[0], angle.shape[0]))
  #return c5x, angle, speed, filters, hdf5_camera
  return c5x, angle, speed, filters, hdf5_camera, gas, gear_choice, brake

"""
   c5x::      [(0, 44792, <HDF5 dataset "X": shape (44792, 3, 160, 320), type "|u1">), 
               (44792, 62969, <HDF5 dataset "X": shape (18177, 3, 160, 320), type "|u1">)
              ] 
   angle::    [ 48.  48.  48. ...,  24.  24.  24.]
   speed::    [ 0.  0.  0. ...,  0.  0.  0.] 
   filter::   [    2     3     4 ..., 62967 62968 62969]
   hdf5_cam:: [<HDF5 file "2016-06-02--21-39-29.h5" (mode r)>, 
               <HDF5 file "2016-06-08--11-46-01.h5" (mode r)>

   Memory calculation per batch: 256*3*160*320 = 39,321,600   => approx (39Mb)

   Total Parameters= ???

"""


first = True


def datagen(filter_files, time_len=1, batch_size=256, ignore_goods=False):
  """
  Parameters:
  -----------
  leads : bool, should we use all x, y and speed radar leads? default is false, uses only x
  """
  global first
  assert time_len > 0
  filter_names = sorted(filter_files)

  logger.info("Loading {} hdf5 buckets.".format(len(filter_names)))
  c5x, angle, speed, filters, hdf5_camera, gas, gear_choice, brake = concatenate(filter_names, time_len=time_len)
  #c5x, angle, speed, filters, hdf5_camera = concatenate(filter_names, time_len=time_len)
  filters_set = set(filters)
  logger.info("camera files {}".format(len(c5x)))

  X_batch = np.zeros((batch_size, time_len, 3, 160, 320), dtype='uint8')
  angle_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
  speed_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
  gas_batch   = np.zeros((batch_size, time_len, 1), dtype='float32')
  gear_batch  = np.zeros((batch_size, time_len, 1), dtype='float32')
  brake_batch = np.zeros((batch_size, time_len, 1), dtype='float32')

  while True:
    try:
      t = time.time()

      count = 0
      start = time.time()
      while count < batch_size:
        if not ignore_goods:
          i = np.random.choice(filters)
          # check the time history for goods
          good = True
          for j in (i-time_len+1, i+1):
            if j not in filters_set:
              good = False
          if not good:
            continue

        else:
          i = np.random.randint(time_len+1, len(angle), 1)

        # GET X_BATCH
        # low quality loop
        for es, ee, x in c5x:
          if i >= es and i < ee:
            X_batch[count] = x[i-es-time_len+1:i-es+1]
            break

        angle_batch[count] = np.copy(angle[i-time_len+1:i+1])[:, None]
        speed_batch[count] = np.copy(speed[i-time_len+1:i+1])[:, None]
        gas_batch[count]   = np.copy(gas[i-time_len+1:i+1])[:, None]
        gear_batch[count]  = np.copy(gear_choice[i-time_len+1:i+1])[:, None]
        brake_batch[count] = np.copy(brake[i-time_len+1:i+1])[:, None]

        count += 1

      # sanity check
      assert X_batch.shape == (batch_size, time_len, 3, 160, 320)

      logging.debug("load image : {}s".format(time.time()-t))
      print("%5.2f ms" % ((time.time()-start)*1000.0))

      if first:
        print ("X", X_batch.shape)
        print ("angle", angle_batch.shape)
        print ("speed", speed_batch.shape)
        print ("gas", gas_batch.shape)
        print ("gear", gear_batch.shape)
        print ("brake", brake_batch.shape)
        first = False

      yield (X_batch, angle_batch, speed_batch, gas_batch, gear_batch, brake_batch)
      #yield (X_batch, angle_batch, speed_batch)


    except KeyboardInterrupt:
      raise
    except:
      traceback.print_exc()
      pass

