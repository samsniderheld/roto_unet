###file for generating augmented data###
import argparse
import os
import glob

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
import torch
import torchvision.transforms as T
from tqdm import tqdm

from utils import *

def parse_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
    """
    desc = "A data augmentation algorithm to generate lines, depth, and transformations"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument('--input_data_dir', type=str, default="train_A/", help='The directory for input data')
    parser.add_argument('--style_data_dir', type=str, default="train_B/", help='The directory for the style data')
    parser.add_argument("--output_dir", type=str, default="dataset/", help="The directory for all the output results.")
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')

    return parser.parse_args()


def transform(files, args, name):
    """
    Transforms the given files using random augmentations and saves the results to disk.

    Parameters:
    files (list): A list of file paths to transform.
    name (str): The name to use when saving the transformed files.

    Returns:
    tuple: The transformed images.
    """
    idx = np.random.randint(len(files))

    rand_x = np.random.random()
    rand_y = np.random.random()

    crop_val = np.random.uniform(.5,.9)
    rand_crop = np.random.random()

    replace_val = tf.constant([1.0,1.0,1.0])

    rand_rot = tf.random.uniform(shape=[1], minval=-1., maxval=1.)
    rand_crop = tf.random.uniform(shape=[1], minval=800., maxval=1024.)

    img_raw_a = tf.io.read_file(files[idx][0])
    img_raw_b = tf.io.read_file(files[idx][1])

    img_a = tf.io.decode_image(img_raw_a)
    img_a = tf.image.resize(img_a,(1024,1024))
    img_a = tfa.image.rotate(img_a, rand_rot,fill_mode='reflect')

    if rand_x > .5:
      img_a = tf.image.flip_left_right(img_a)
    if rand_y > .5:
      img_a = tf.image.flip_up_down(img_a)


    img_a = tf.image.resize(img_a,(1024,1024)).numpy()
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2RGB)

    img_c = create_hed(img_a,512,512)
    img_d = create_midas(img_a)

    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)

    img_b = tf.io.decode_image(img_raw_b)
    img_b = tf.image.resize(img_b,(1024,1024))
    img_b = tfa.image.rotate(img_b, rand_rot,fill_mode='reflect')

    if rand_x > .5:
      img_b = tf.image.flip_left_right(img_b)
    if rand_y > .5:
      img_b = tf.image.flip_up_down(img_b)


    img_b = tf.image.resize(img_b,(1024,1024)).numpy()
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)



    cv2.imwrite(f"{args.output_dir}train_A/{name:04d}.jpg",img_a)
    cv2.imwrite(f"{args.output_dir}lines/{name:04d}.jpg",img_c)
    cv2.imwrite(f"{args.output_dir}depth/{name:04d}.jpg",img_d)
    cv2.imwrite(f"{args.output_dir}train_B/{name:04d}.jpg",img_b)
    return img_a,img_b


def main():

    args = parse_args()

    input_path = os.path.join(args.input_data_dir,"*")
    style_path = os.path.join(args.style_data_dir,"*")

    output_dir = args.output_dir 
    
    directories = [
        os.path.join(output_dir, "train_A"),
        os.path.join(output_dir, "lines"),
        os.path.join(output_dir, "depth"),
        os.path.join(output_dir, "train_B"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


    files_A = sorted(glob.glob(input_path), key=natural_keys)
    files_B = sorted(glob.glob(style_path), key=natural_keys)

    num = len(files_B)
    files = list(zip(files_A[:num],files_B[:num]))

    for i in tqdm(range(0,args.num_samples)):
        transform(files,args,i)

if __name__ == '__main__':
    main()