# Driving State Prediction using LRCN
In this project, we are currently extending on the prior art for predictive modeling of the other driving states like gas pedal (accelerator), clutch, gear lever, brake pedal press using multi-sensor integration for modeling of these states. We are implementing a LRCN (LSTM + CNN) model for the driving state prediction. The publicly available driving dataset consisting of 7.25 hours of highway rich quality video feed with other sensor data like speed, acceleration, 3D LIDAR along with car driving control variable data for the training.

## Comma.ai's paper

[Learning a Driving Simulator](http://arxiv.org/abs/1608.01230)

## Comma.ai and Udacity online released driving dataset

7 and a quarter hours of largely highway driving. Enough to train what we had in [Bloomberg](http://www.bloomberg.com/features/2015-george-hotz-self-driving-car/).

Udacity has released 223 Gb of driving dataset which consist of 2 set of datasets.

## Examples

We present two Machine Learning Experiments to show
possible ways to use this dataset:


<img src="./images/selfsteer.gif">

[Training a steering angle predictor](SelfSteering.md)


## Requirements
[anaconda](https://www.continuum.io/downloads)  
[tensorflow-0.9](https://github.com/tensorflow/tensorflow)  
[keras-1.0.6](https://github.com/fchollet/keras)  
[cv2](https://anaconda.org/menpo/opencv3)
