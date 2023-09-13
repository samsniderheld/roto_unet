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


def perceptual_loss_builder(layers_list):
    
    # Load the VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)
    
    # Get the output of the specified layers
    if not isinstance(layers_list, list):
        layers_list = [layers_list]

    outputs = [base_model.get_layer(layer).output for layer in layers_list]
    
    # Build the intermediate model
    intermediate_model = Model(inputs=base_model.input, outputs=outputs)
    
    # Define the perceptual loss function
    def perceptual_loss(y_true, y_pred):

        # Convert the images to 3 channels
        y_true = tf.repeat(y_true, 3, axis=-1)
        y_pred = tf.repeat(y_pred, 3, axis=-1)

        # Resize images to fit VGG16 input size
        y_true = tf.image.resize(y_true, [224, 224])
        y_pred = tf.image.resize(y_pred, [224, 224])

        # Get the features for the true and predicted images
        y_true_features = intermediate_model(y_true)
        y_pred_features = intermediate_model(y_pred)

        if len(layers_list) == 1:
            y_true_features = [y_true_features]
            y_pred_features = [y_pred_features]
        
        # Compute the perceptual loss as the average mse across all specified layers
        loss = sum(tf.reduce_mean(tf.square(y_true_f - y_pred_f)) for y_true_f, y_pred_f in zip(y_true_features, y_pred_features))
        loss /= len(layers_list)
        
        return loss
    
    return perceptual_loss
