from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras import backend as K
from keras.layers.merge import concatenate

def preprocess_input(x):
    x = x/255
    return x

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #return K.sum((y_true_f - y_pred_f)**2)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj, 
                     dropout=0.0, 
                     batch_norm=True
                     ):
    '''
    https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
    '''
    axis=3

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    if batch_norm is True:
        conv_1x1 = BatchNormalization(axis=axis)(conv_1x1)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)
    if batch_norm is True:
        conv_3x3 = BatchNormalization(axis=axis)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)
    if batch_norm is True:
        conv_5x5 = BatchNormalization(axis=axis)(conv_5x5)

    pool_proj = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

    conv = Conv2D(2*filters_1x1, (3, 3), padding='same')(output)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)

    return conv


def UInceptionNet_binary_model(dropout_val=0.2):
    inputs = Input((224, 224, 3))
    axis = 3
    filters = 8

    conv_224 = inception_module(inputs, filters, 1, filters, 1, filters, filters)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = inception_module(pool_112, 2*filters, 8, 2*filters, 8, 2*filters, 2*filters)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = inception_module(pool_56, 4*filters, 16, 4*filters, 16, 4*filters, 4*filters)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = inception_module(pool_28, 8*filters, 32, 8*filters, 32, 8*filters, 8*filters)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = inception_module(pool_14, 16*filters, 64, 16*filters, 64, 16*filters, 16*filters)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = inception_module(pool_7, 32*filters, 128, 32*filters, 128, 32*filters, 32*filters)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = inception_module(up_14, 16*filters, 64, 16*filters, 64, 16*filters, 16*filters)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = inception_module(up_28, 8*filters, 8*filters, 32, 8*filters, 32, 8*filters, 8*filters)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = inception_module(up_56, 4*filters, 4*filters, 16, 4*filters, 16, 4*filters, 4*filters)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = inception_module(up_112, 2*filters, 2*filters, 8, 2*filters, 8, 2*filters, 2*filters)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = inception_module(up_224, filters, filters, 1, filters, 1, filters, filters, dropout_val)

    conv_final = Conv2D(1, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="U-NET")

    return model