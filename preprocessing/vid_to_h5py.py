
from __future__ import print_function

# import imageio
import os
import h5py
from glob import glob

import subprocess
from scipy.misc import imread, imresize

"""
Taken help from: https://github.com/EderSantana/seya/tree/master/seya/preprocessing
"""
def vid_to_hdf5(hdf5file, filepath,
                           ext='*.avi',
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
    with h5py.File(hdf5file, 'w') as h5:
        # create datasets
        X = h5.create_dataset('data', (len(files), len(frame_range),
                              rlen, clen, 3), dtype='f')
        if label_callback is not None:
                label_size = label_callback('test', return_len=True)
                y = h5.create_dataset('labels', (len(files), label_size), dtype='f')

        for c1, f in enumerate(files):
            process = subprocess.Popen('mkdir {}'.format(f[:-4]), shell=True, stdout=subprocess.PIPE)
            process.wait()
            cmd = converter + " -i {0} -r 1/1 {1}/%03d.jpg".format(f, f[:-4])
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()

            imgs = glob(os.path.join(f[:-4], '*.jpg'))
            for c2, im in enumerate(imgs[:frame_range[-1]]):
                I = imread(im)[row_range, col_range]
                if I.shape[:2] != img_shape:
                    I = imresize(imread(im), img_shape)
                X[c1, c2] = I

            if label_callback is not None:
                y[c1] = label_callback(f)

            process = subprocess.Popen('rm {}'.format(f[:-4]), shell=True, stdout=subprocess.PIPE)
            process.wait()

    with open(hdf5file+'.txt', 'w') as file_order:
        for f in files:
            file_order.write(f+'\n')
