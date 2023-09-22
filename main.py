"""
This module contains a script for training a U-Net generator with 
progressive growing and perceptual loss. The generator is built, 
compiled, and trained based on the dimension parameter provided 
through the command line arguments.

The script performs the following operations:
1. Parses command line arguments to obtain configurations such as input 
   data directory, line data directory, depth data directory, style 
   data directory, and the dimensions to train the generator at.
2. Depending on the dimension argument, it sets up the generator with 
   the appropriate configuration and compiles it using the Adam 
   optimizer and a perceptual loss function.
3. Creates a data generator instance with the appropriate batch size 
   and dimensions.
4. Runs the training loop for a predefined number of epochs. In each 
   epoch, it trains the generator for one epoch, retrieves a random 
   batch of data, and gets a prediction from the generator. It then 
   displays the input sample, lines, depth, target output, and prediction 
   side by side using matplotlib.
5. Saves the trained generator model to a predetermined directory.

Usage:
    Run this script from the command line using the following format:
    python main.py --dims DIMENSIONS
    

"""
import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from data.data_generator import DataGenerator
from model.model import create_generator, add_layers_to_unet
# from model.loss import perceptual_loss

from model.loss import perceptual_loss_builder

def parse_args():
    """Parse the command line arguments."""
    desc = "An attention based unet with progressive growing and perceptual_loss"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--data_dir', type=str, default="dataset",
        help='The directory for data'
    )
    parser.add_argument(
        '--dims', type=int, default=256,
        help='The dimensions to train at'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='The dimensions to train at'
    )
    parser.add_argument(
        '--use_depth', action='store_true',
        help='enable depth map condition'
    )
    parser.add_argument(
        '--perceptual_loss_layers', type=str, nargs='+', default=['block3_conv3', 'block2_conv2'],
        help='The dimensions to train at'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='The number of epochs to train for'
    )
    parser.add_argument(
        '--saved_models', type=str, default="saved_models",
        help='The directory where all the save model weights live.'
    )

    return parser.parse_args()


args = parse_args()

os.makedirs(args.saved_models, exist_ok=True)

perceptual_loss = perceptual_loss_builder(args.perceptual_loss_layers)

if args.dims == 256:
    generator = create_generator((256, 256, 1),args.use_depth)
    generator.compile(optimizer="adam", loss=perceptual_loss)
    generator.summary()
    data_gen = DataGenerator(args.data_dir, batch_size=args.batch_size, dims=256)

elif args.dims == 512:
    generator = create_generator((256, 256, 1),args.use_depth)
    generator.load_weights(f"{args.saved_models}/256")
    generator = add_layers_to_unet(generator)
    generator.compile(optimizer="adam", loss=perceptual_loss)
    generator.summary()
    data_gen = DataGenerator(args.data_dir, batch_size=args.batch_size, dims=512)

elif args.dims == 1024:
    generator = create_generator((256, 256, 1),args.use_depth)
    generator = add_layers_to_unet(generator)
    generator.load_weights(f"{args.saved_models}/512")
    generator = add_layers_to_unet(generator, 1024)
    generator.compile(optimizer="adam", loss=perceptual_loss)
    generator.summary()
    data_gen = DataGenerator(args.data_dir, batch_size=args.batch_size, dims=1024)


for i in range(args.epochs):
    print(f"training epoch {i}")
    generator.fit(data_gen, epochs=1)

    batch = data_gen.__getitem__(random.randrange(0, data_gen.count))
    sample = np.squeeze(batch[0][0][0])
    lines = np.squeeze(batch[0][1][0])
    if(args.use_depth):
        depth = np.squeeze(batch[0][2][0])
    out_sample_2 = np.squeeze(batch[1][0])

    if(args.use_depth):
        prediction = generator([
            np.expand_dims(sample, 0),
            np.expand_dims(lines, 0),
            np.expand_dims(depth, 0)
        ])
    else:
        prediction = generator([
            np.expand_dims(sample, 0),
            np.expand_dims(lines, 0),
        ])
        
    prediction = np.squeeze(prediction)

    plt.figure(figsize=(10, 40))

    if(args.use_depth):
        plt.imshow(np.hstack([sample, out_sample_2, lines, depth, prediction]), cmap="gray")
    else:
        plt.imshow(np.hstack([sample, out_sample_2, lines, prediction]), cmap="gray")
    plt.show()

checkpoint_path = f"{args.saved_models}/{args.dims}/"
generator.save(checkpoint_path)
