"""
This module contains functionality to set up a perceptual loss function 
using a pre-trained VGG16 model. The perceptual loss function computes the 
mean squared error loss between feature maps extracted from different layers 
of the VGG16 model, providing a measure of perceptual similarity between 
predicted and ground truth images.


"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as K


# Load a pre-trained VGG16 model and set up as feature extractors
vgg = VGG16(weights='imagenet', include_top=False)
vgg_1 = Model(inputs=vgg.inputs, outputs=vgg.get_layer('block3_conv3').output)
vgg_2 = Model(inputs=vgg.inputs, outputs=vgg.get_layer('block2_conv2').output)

# Ensure the VGG model is not trainable
for layer in vgg.layers:
    layer.trainable = False


def perceptual_loss(y_true, y_pred):
    """
    Computes the perceptual loss between the true and predicted images based
    on the feature maps extracted from a pre-trained VGG16 model.

    Parameters:
    - y_true (Tensor): Ground truth image tensor.
    - y_pred (Tensor): Predicted image tensor.

    Returns:
    - loss (Tensor): The computed perceptual loss.
    """
    # Convert the images to 3 channels
    y_true = tf.repeat(y_true, 3, axis=-1)
    y_pred = tf.repeat(y_pred, 3, axis=-1)

    # Resize images to fit VGG16 input size
    y_true = tf.image.resize(y_true, [224, 224])
    y_pred = tf.image.resize(y_pred, [224, 224])

    # Extract the feature maps
    y_true_features_1 = vgg_1(y_true)
    y_pred_features_1 = vgg_1(y_pred)

    y_true_features_2 = vgg_2(y_true)
    y_pred_features_2 = vgg_2(y_pred)

    # Calculate the MSE between the feature maps
    loss_1 = tf.reduce_mean(tf.square(y_true_features_1 - y_pred_features_1))
    loss_2 = tf.reduce_mean(tf.square(y_true_features_2 - y_pred_features_2))

    # Calculate the overall loss as the average of the two losses
    loss = (loss_1 + loss_2) / 2

    return loss
