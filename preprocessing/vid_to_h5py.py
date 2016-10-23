
from __future__ import print_function

# import imageio
import os
import h5py
import cv2
import numpy as np
import sys
import skvideo.io
from glob import glob

import subprocess
from scipy.misc import imread, imresize

"""
Taken help from: https://github.com/EderSantana/seya/tree/master/seya/preprocessing
"""
def vid_to_hdf5(hdf5file='test.h5', filepath='/home/rahulsingh/research/preprocessing/',
                           ext='*.mp4',
                           img_shape=(224, 224),
                           frame_range=range(0, 20),
                           row_range=slice(120-112, 120+112),
                           col_range=slice(160-112, 160+112),
                           label_callback=None,
                           converter='avconv'):

    """
    Convert all the videos of a folder to images and dump the images to an hdf5
    file.
    Parameters:
    -----------
    hdf5file: str, path to output hdf5 file
    filepath: str, path to folder with videos
    ext: str, video extensions (default *.avi)
    img_shape: tuple, (row, col) size of the image crop of video
    frame_range: list, desired frames of the video
    row_range: slice, slice of the image rows
    col_rance: slice, slice of the image cols
    Results:
    --------
    An hdf5 file with videos stored as array
    """
    rlen = row_range.stop - row_range.start
    clen = col_range.stop - col_range.start
    files = glob(os.path.join(filepath, ext))
    f='/home/rahulsingh/research/preprocessing/small.mp4'
    c1=0
    with h5py.File(hdf5file, 'w') as h5:
        process = subprocess.Popen('mkdir {}'.format(f[:-4]), shell=True, stdout=subprocess.PIPE)
        process.wait()
        cmd = converter + " -i {0} -r 1/1 {1}/%03d.jpg".format(f, f[:-4])
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()

        imgs = glob(os.path.join(f[:-4], '*.jpg'))
        X = h5.create_dataset('data', (len(imgs), rlen, clen, 3), dtype='f')
        for c2, im in enumerate(imgs[:]):#frame_range[-1]]):
            #print(frame_range[-1])
            I = imread(im)[row_range, col_range]
            if I.shape[:2] != img_shape:
                I = imresize(imread(im), img_shape)
            print(c2)
            X[c2] = I

        process = subprocess.Popen('rm -r {}'.format(f[:-4]), shell=True, stdout=subprocess.PIPE)
        process.wait()


    with open(hdf5file+'.txt', 'w') as file_order:
        for f in files:
            file_order.write(f+'\n')



"""
    Apparently opencv 'Videowriter' has a bit of codec issue problems which 
    according to some websites gets solved by trial and error or by configuring
    ffmped.
    Links: 
    Configuring ffmped: http://stackoverflow.com/questions/11444926/videocapture-is-not-working-in-opencv-2-4-2/11465097#11465097
    A blog: www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/

    The issue can be solved as temporarily using scikit-video library. Heres the link
    Git link: https://github.com/ContinuumIO/anaconda-issues/issues/121
"""
def cam_writer(device=0):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    # Define the codec and create VideoWriter object

    while(cap.isOpened()):
      ret, frame = cap.read()
      if ret==True:
        #frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      else:
        print("Problem with reading video frames")
        break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def vid_view():
    #help(skvideo.io.VideoCapture)
    cap = skvideo.io.VideoCapture(str("./test.mp4"))  # u = unicode(s, "utf-8")
    ret, frame1 = cap.read()
    imshow('rah', frame)


if __name__ == '__main__': 
    #cam_writer()
    #vid_view()
    vid_to_hdf5()