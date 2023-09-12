"""
A script that utilizes a trained neural network generator model for data augmentation. The script reads input data from a specified
directory, processes the data to create HED (Holistic Edge Detection) and MiDaS (depth estimation) augmentations, and then uses a
pre-trained generator to create a new augmentation based on the input and the HED and MiDaS augmentations. The resulting augmented
data is saved to a specified output directory. 

Usage:
    python script_name.py --input_dir [input_directory] --output_dir [output_directory] --dims [dimensions] --saved_models [saved_models_directory]

Arguments:
    --input_dir: The directory for input data. Default is "train_A/".
    --output_dir: The directory for all output results. Default is "outputs/".
    --dims: The dimensions to render at. Default is 512.
    --saved_models: The directory where all the saved model weights reside. Default is "saved_models".
"""

import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model.model import create_generator, add_layers_to_unet
from utils import create_hed, create_midas

def parse_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
    """
    desc = "A data augmentation algorithm to generate lines, depth, and transformations"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument(
        '--input_dir', type=str, default='train_A/', 
        help='The directory for input data')
    parser.add_argument(
        '--output_dir', type=str, default='outputs/', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--dims', type=int, default=512, 
        help='The Dimensions to Render at')
    parser.add_argument(
        '--saved_models', type=str, default="saved_models",
        help='The directory where all the save model weights live.'
    )

    return parser.parse_args()

args = parse_args()

os.makedirs(args.output_dir, exist_ok=True)

if args.dims == 256:
    generator = create_generator((256, 256, 1))
    generator.load_weights(f"{args.saved_models}/256")


elif args.dims == 512:
    generator = create_generator((256, 256, 1))
    generator = add_layers_to_unet(generator)
    generator.load_weights(f"{args.saved_models}/512")

elif args.dims == 1024:
    generator = create_generator((256, 256, 1))
    generator = add_layers_to_unet(generator)
    generator = add_layers_to_unet(generator, 1024)
    generator.load_weights(f"{args.saved_models}/1024")

inputs  = sorted(glob.glob(args.input_dir))

for i in tqdm(range(0,len(inputs))):

    sample = cv2.imread(inputs[i])
    lines = create_hed(sample,512,512)
    depth = create_midas(sample)

    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    lines = cv2.cvtColor(lines, cv2.COLOR_BGR2GRAY)

    sample = sample/255
    sample = cv2.resize(sample,(args.dims,args.dims))

    lines = lines/255
    lines = cv2.resize(lines,(args.dims,args.dims))

    depth = depth/255
    depth = cv2.resize(depth,(args.dims,args.dims))


    prediction = generator([
        np.expand_dims(sample, 0),
        np.expand_dims(lines, 0),
        np.expand_dims(depth, 0)
    ])


    prediction = np.squeeze(prediction)*255

    cv2.imwrite(args.output_dir + "/" +f"{i : 04d}.png", prediction)