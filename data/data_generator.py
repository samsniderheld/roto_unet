"""
This module contains a data generator class `DataGenerator` which extends 
tf.keras.utils.Sequence. It is designed to generate batches of data from 
specified directories, facilitating the training of a neural network 
using Keras.

Usage:
    Instantiate an object of the DataGenerator class by providing the 
    necessary parameters including the data directories and batch size.

    Example:
    data_gen = DataGenerator(dir='./path/to/data/', batch_size=4, dims=1024)

"""
import os
import tensorflow as tf
import numpy as np
import glob
import random
import cv2


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""
    
    def __init__(self, dir='dataset', batch_size=4, dims=1024):
        """Initialization"""
        self.input_dir = os.path.join(dir,'train_A/*.jpg')
        self.lines_dir = os.path.join(dir,'lines/*.jpg')
        self.depth_dir = os.path.join(dir,'depth/*.jpg')
        self.output_dir = os.path.join(dir,'train_B/*.jpg')
        self.batch_size = batch_size
        self.dims = dims
        self.inputs = sorted(glob.glob(self.input_dir))
        self.lines = sorted(glob.glob(self.lines_dir))
        self.depth = sorted(glob.glob(self.depth_dir))
        self.outputs = sorted(glob.glob(self.output_dir))
        self.count = self.__len__()
        print("Number of all samples =", len(self.inputs))

    def __len__(self):
        """Denotes the number of batches per epoch"""
        self.num_batches = int(np.floor(len(self.inputs) / self.batch_size))
        return self.num_batches

    def __getitem__(self, index):
        """Retrieve batch at index"""
        X = self.__data_generation(index)
        return X

    def on_epoch_end(self):
        """Actions to do on epoch end"""
        pass

    def __data_generation(self, idx):
        """Generates data containing batch_size samples"""

        # Define batch file lists
        input_files = self.inputs[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        lines_files = self.lines[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        depth_files = self.depth[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        output_files = self.outputs[idx*self.batch_size:idx*self.batch_size+self.batch_size]

        # Initialize data arrays
        X = np.empty((self.batch_size, self.dims, self.dims))
        L = np.empty((self.batch_size, self.dims, self.dims))
        D = np.empty((self.batch_size, self.dims, self.dims))
        Y = np.empty((self.batch_size, self.dims, self.dims))

        # Load and preprocess data
        for i,_ in enumerate(input_files):
            img_0 = cv2.imread(input_files[i])
            img_0 = cv2.resize(img_0, (self.dims, self.dims))
            X[i] = img_0[:, :, 0] / 255

            lines = cv2.imread(lines_files[i])
            lines = cv2.resize(lines, (self.dims, self.dims))
            L[i] = lines[:, :, 0] / 255

            depth = cv2.imread(depth_files[i])
            depth = cv2.resize(depth, (self.dims, self.dims))
            D[i] = depth[:, :, 0] / 255

            out_img_0 = cv2.imread(output_files[i])
            out_img_0 = cv2.resize(out_img_0, (self.dims, self.dims))
            Y[i] = out_img_0[:, :, 0] / 255

        # Expand dimensions to fit the model input
        X = np.expand_dims(X, axis=3)
        L = np.expand_dims(L, axis=3)
        D = np.expand_dims(D, axis=3)
        Y = np.expand_dims(Y, axis=3)

        return ([X, L, D], Y)
