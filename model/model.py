"""
This module defines a set of functions to create a generator model based on the U-Net architecture,
enhanced with self-attention mechanisms and the possibility to increase its resolution by adding
additional encoder and decoder layers.

The U-Net model is commonly used for semantic segmentation in the field of computer vision.
"""

import tensorflow as tf
from tensorflow.keras.layers import Add, Activation, Conv2D, Input, MaxPooling2D, Multiply, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def conv2d_block(input_tensor, num_filters):
    """
    Creates a block containing two convolutional layers each followed by a ReLU activation function.

    Parameters:
    - input_tensor (Tensor): The input tensor for the block.
    - num_filters (int): The number of filters for the convolutional layers.

    Returns:
    - x (Tensor): The output tensor from the block.
    """
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    return x


def self_attention(input, name=None):
    """
    Implements a self-attention mechanism over the input tensor.

    Parameters:
    - input (Tensor): The input tensor for the self-attention mechanism.
    - name (str, optional): An optional name for the added layer.

    Returns:
    - out (Tensor): The output tensor from the self-attention mechanism.
    """
    q = Conv2D(1, (1, 1), padding='same', activation='relu')(input)
    k = Conv2D(1, (1, 1), padding='same', activation='relu')(input)
    v = Conv2D(1, (1, 1), padding='same', activation='relu')(input)

    qk = Multiply()([q, k])
    att = Activation('softmax')(qk)
    out = Multiply()([v, att])
    out = Add(name=name)([out, input])

    return out


def create_generator(input_shape):
    """
    Creates a U-Net model augmented with self-attention mechanisms.

    Parameters:
    - input_shape (tuple): The shape of the input tensor.

    Returns:
    - model (Model): The constructed U-Net model.
    """
    inputs = Input(input_shape, name="image")
    condition_input = Input(input_shape, name="condition_image")
    condition_input_2 = Input(input_shape, name="condition_image_2")

    x = Add()([inputs, condition_input, condition_input_2])
    
    # Downsampling path
    c1 = conv2d_block(x, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    p1 = self_attention(p1)

    c2 = conv2d_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)

    p2 = self_attention(p2)

    c3 = conv2d_block(p2, 128)

    # Self attention
    c3 = self_attention(c3,name="top_half")


    bottle_neck_input = tf.image.resize(condition_input,[64,64])
    bottle_neck_input_2 = tf.image.resize(condition_input_2,[64,64])
    bottle_neck = Add()([c3,bottle_neck_input,bottle_neck_input_2])

    # Upsampling path
    u4 = UpSampling2D((2, 2),name="bottom_half")(bottle_neck)
    u4 = concatenate([u4, c2])
    c4 = conv2d_block(u4, 64)

    c4 = self_attention(c4)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = conv2d_block(u5, 32)

    c5 = self_attention(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = tf.keras.Model(inputs=[inputs,condition_input,condition_input_2], outputs=[outputs])

    return model


# TODO: fix this bug and add depth conditional layer
def add_layers_to_unet(model, new_dim=512):
    """
    Adds additional encoder and decoder layers to a U-Net model to accommodate for an increased resolution.

    Parameters:
    - model (Model): The initial U-Net model.
    - new_dim (int): The new dimension size for the input tensors.

    Returns:
    - updated_model (Model): The updated U-Net model with additional layers.
    """

    layers = model.layers
    
    model = tf.keras.Model(inputs=[layers[3].input,layers[1].input,layers[2].input], outputs=layers[-1].output)


    inputs = Input((new_dim,new_dim,1))
    condition_input = Input((new_dim,new_dim,1))
    condition_input_2 = Input((new_dim,new_dim,1))

    resize_dim = int(new_dim/2)

    added_inputs = Add()([inputs,condition_input, condition_input_2])

    c1 = conv2d_block(added_inputs, 16)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Conv2D(1, (1, 1), activation='relu')(p1)

    bottle_neck_condition_input = tf.image.resize(condition_input,[resize_dim,resize_dim])
    bottle_neck_condition_input_2 = tf.image.resize(condition_input_2,[resize_dim,resize_dim])
    x = model([p1,bottle_neck_condition_input,bottle_neck_condition_input_2])

    c2 = self_attention(x)

    u1 = UpSampling2D((2, 2))(c2)
    u1 = concatenate([u1, c1])
    c3 = conv2d_block(u1, 16)

    outputs = Conv2D(1, (1, 1), activation='tanh')(c3)

    # Construct the updated model
    updated_model = tf.keras.Model(inputs=[inputs,condition_input,condition_input_2], outputs=[outputs])


    return updated_model


