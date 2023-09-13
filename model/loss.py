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
    outputs = [base_model.get_layer(layer).output for layer in layers_list]
    
    # Build the intermediate model
    intermediate_model = Model(inputs=base_model.input, outputs=outputs)
    
    # Define the perceptual loss function
    def perceptual_loss(y_true, y_pred):
        # Get the features for the true and predicted images
        y_true_features = intermediate_model(y_true)
        y_pred_features = intermediate_model(y_pred)
        
        # Compute the perceptual loss as the average mse across all specified layers
        loss = sum(tf.reduce_mean(tf.square(y_true_f - y_pred_f)) for y_true_f, y_pred_f in zip(y_true_features, y_pred_features))
        loss /= len(layers_list)
        
        return loss
    
    return perceptual_loss


# # Load a pre-trained VGG16 model and set up as feature extractors
# def get_vgg_models(perception_layers):

#     # Ensure the VGG model is not trainable

#     vgg = VGG16(weights='imagenet', include_top=False)
#     for layer in vgg.layers:
#         layer.trainable = False

#     models = []

#     for p_layer in perception_layers:
#         models.append(Model(inputs=vgg.inputs, outputs=vgg.get_layer(p_layer).output))

#     return models

# # vgg = VGG16(weights='imagenet', include_top=False)
# # vgg_1 = Model(inputs=vgg.inputs, outputs=vgg.get_layer('block3_conv3').output)
# # vgg_2 = Model(inputs=vgg.inputs, outputs=vgg.get_layer('block2_conv2').output)

# # # Ensure the VGG model is not trainable
# # for layer in vgg.layers:
# #     layer.trainable = False


# def perceptual_loss(y_true, y_pred):
#     """
#     Computes the perceptual loss between the true and predicted images based
#     on the feature maps extracted from a pre-trained VGG16 model.

#     Parameters:
#     - y_true (Tensor): Ground truth image tensor.
#     - y_pred (Tensor): Predicted image tensor.

#     Returns:
#     - loss (Tensor): The computed perceptual loss.
#     """
#     # Convert the images to 3 channels
#     y_true = tf.repeat(y_true, 3, axis=-1)
#     y_pred = tf.repeat(y_pred, 3, axis=-1)

#     # Resize images to fit VGG16 input size
#     y_true = tf.image.resize(y_true, [224, 224])
#     y_pred = tf.image.resize(y_pred, [224, 224])

#     # Extract the feature maps
#     y_true_features_1 = vgg_1(y_true)
#     y_pred_features_1 = vgg_1(y_pred)

#     y_true_features_2 = vgg_2(y_true)
#     y_pred_features_2 = vgg_2(y_pred)

#     # Calculate the MSE between the feature maps
#     loss_1 = tf.reduce_mean(tf.square(y_true_features_1 - y_pred_features_1))
#     loss_2 = tf.reduce_mean(tf.square(y_true_features_2 - y_pred_features_2))

#     # Calculate the overall loss as the average of the two losses
#     loss = (loss_1 + loss_2) / 2

#     return loss
