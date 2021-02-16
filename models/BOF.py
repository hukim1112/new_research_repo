import tensorflow as tf
from tensorflow.keras.layers import Conv2D

def block(num_filter, input_shape):
    block1 = tf.keras.Sequential()
    block1.add(Conv2D(num_filter,(1,1),strides=1,activation='relu'))
    block1.add(Conv2D(num_filter,(3,3),strides=2,padding="same",activation='relu'))
    block1.add(Conv2D(num_filter,(1,1),strides=1,activation='relu'))
    block2 = tf.keras.Sequential()
    block1.add(Conv2D(num_filter,(1,1),strides=1,activation='relu'))
    block1.add(Conv2D(num_filter,(1,1),strides=1,activation='relu'))
    block1.add(Conv2D(num_filter,(1,1),strides=1,activation='relu'))
    block3 = tf.keras.Sequential()
    block1.add(Conv2D(num_filter,(1,1),strides=1,activation='relu'))
    block1.add(Conv2D(num_filter,(1,1),strides=1,activation='relu'))
    block1.add(Conv2D(num_filter,(1,1),strides=1,activation='relu'))

    input_layer = tf.keras.Input(shape=input_shape)
    residual = Conv2D(num_filter,(1,1),strides=2,activation='relu')(input_layer)
    x = block1(input_layer)
    x = x + residual

    residual = Conv2D(num_filter, (1,1),strides=1,activation='relu')(x)
    x = block2(x)
    x = x + residual

    residual = Conv2D(num_filter, (1,1),strides=1,activation='relu')(x)
    x = block3(x)
    x = x + residual
    resnet_block = tf.keras.Model(inputs=input_layer, outputs=x)
    return resnet_block
