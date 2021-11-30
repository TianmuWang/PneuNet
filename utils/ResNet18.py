import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    """
    conv2d -> batch normalization -> relu activation
    """
    x = layers.Conv2D(nb_filter, kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
 
 
def shortcut(input, residual): # here the data is shaped as (N, H, W, C)
 
    input_shape = keras.backend.int_shape(input)
    residual_shape = keras.backend.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3] # to tell if channel of X equals the channel fo residual 
 
    identity = input
    # if equal_channel == 0, where channel of X does not match channel of residual
    if stride_width > 1 or stride_height > 1 or not equal_channels: # use (1,1) sized kernel to make compliant
        identity = layers.Conv2D(filters=residual_shape[3],
                           kernel_size=(1, 1),
                           strides=(stride_width, stride_height),
                           padding="valid",
                           kernel_regularizer=keras.regularizers.l2(0.0001))(input)
 
    return layers.add([identity, residual])
 
 
def basic_block(nb_filter, strides=(1, 1)):
    """
    基本的ResNet building block，适用于ResNet-18和ResNet-34.
    """
    def f(input):
 
        conv1 = conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
        residual = conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))
 
        return shortcut(input, residual)
 
    return f

def deeper_block(nb_filter, strides=(1, 1)):



    # return f
    pass
 
 
def residual_block(nb_filter, repetitions, is_first_layer=False):
    """
    构建每层的residual模块，对应论文参数统计表中的conv2_x -> conv5_x
    """
    def f(input):
        for i in range(repetitions):
            strides = (1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2)
            input = basic_block(nb_filter, strides)(input)
        return input
 
    return f
 
 
def resnet_18(input_shape=(224,224,3), nclass=1000, Denselayer = False):
    """
    build resnet-18 model using keras with TensorFlow backend.
    :param input_shape: input shape of network, default as (224,224,3)
    :param nclass: numbers of class(output shape of network), default as 1000
    :return: resnet-18 model
    """
    input_ = layers.Input(shape=input_shape)
 
    conv1 = conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
 
    conv2 = residual_block(64, 2, is_first_layer=True)(pool1)
    conv3 = residual_block(128, 2, is_first_layer=True)(conv2)
    conv4 = residual_block(256, 2, is_first_layer=True)(conv3)
    conv5 = residual_block(512, 2, is_first_layer=True)(conv4)
    # conv5 = residual_block(256, 2, is_first_layer=True)(conv4)

    pool2 = layers.GlobalAvgPool2D()(conv5)
    output_ = layers.Dense(nclass, activation='softmax')(pool2)
    if Denselayer:
        model = keras.models.Model(inputs=input_, outputs=output_)
    else:
        model = keras.models.Model(inputs = input_, outputs = conv5)
    # model.summary()
 
    return model
 
if __name__ == '__main__':
    model = resnet_18()
    keras.utils.plot_model(model, 'ResNet-18.png')


