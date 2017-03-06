import os

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

import luna_unet_data2


_DATA_DIR = '../LUNA16/output_unet_data2'
_MODEL_PATH = './unet2d1.hdf5'

_IMAGE_ROWS = 96
_IMAGE_COLS = 96

_IMAGES_MEAN = 0.180
_IMAGES_STD = 0.270

_BATCH_SIZE = 6
_NUM_EPOCHS = 100

_SMOOTH = 1.


K.set_image_dim_ordering('th')  # Theano dimension ordering in this code


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + _SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + _SMOOTH)


def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + _SMOOTH) / (np.sum(y_true_f) + np.sum(y_pred_f) + _SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def _conv(nb_filter, nb_row, nb_col, input):
    ans = Convolution2D(nb_filter, nb_row, nb_col, border_mode='same')(input)
    ans = BatchNormalization(mode=1)(ans)
    ans = Activation('relu')(ans)
    ans = Dropout(0.5)(ans)
    return ans


def get_unet():
    inputs = Input((1, _IMAGE_ROWS, _IMAGE_COLS))

    conv1 = _conv(32, 3, 3, inputs)
    conv1 = _conv(32, 3, 3, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = _conv(64, 3, 3, pool1)
    conv2 = _conv(64, 3, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = _conv(128, 3, 3, pool2)
    conv3 = _conv(128, 3, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = _conv(256, 3, 3, pool3)
    conv4 = _conv(256, 3, 3, conv4)

    up5 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
    conv5 = _conv(128, 3, 3, up5)
    conv5 = _conv(128, 3, 3, conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv2], mode='concat', concat_axis=1)
    conv6 = _conv(64, 3, 3, up6)
    conv6 = _conv(64, 3, 3, conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv1], mode='concat', concat_axis=1)
    conv7 = _conv(32, 3, 3, up7)
    conv7 = _conv(32, 3, 3, conv7)

    conv8 = Convolution2D(1, 1, 1, activation='sigmoid')(conv7)

    model = Model(input=inputs, output=conv8)

    model.compile(optimizer=Adam(lr=1.0e-5),
                  loss=dice_coef_loss, metrics=[dice_coef])

    return model
        

def normalize_images(images):
    return (images - _IMAGES_MEAN) / _IMAGES_STD


def load_data(subsets):
    images, nodule_masks = luna_unet_data2.load_data(subsets)
    images = normalize_images(images)
    return images, nodule_masks
    

def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train, imgs_mask_train = load_data(
        ['subset0', 'subset1', 'subset2',
         'subset3', 'subset4', 'subset5',
         'subset6', 'subset7', 'subset8'])

    imgs_test, imgs_mask_test = load_data(['subset9'])

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # Saving weights to unet2.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint(_MODEL_PATH, monitor='loss', save_best_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights(_MODEL_PATH)
        
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train,
              validation_data = (imgs_test, imgs_mask_test),
              batch_size=_BATCH_SIZE, nb_epoch=_NUM_EPOCHS,
              verbose=1, shuffle=True,
              callbacks=[model_checkpoint])


def test():
    imgs_test, imgs_mask_test = load_data(['subset9'])

    model = get_unet()
    model.load_weights('_MODEL_PATH')
    
    imgs_mask_pred = model.predict(imgs_test)
    dice = []
    for i in range(len(imgs_test)):
        dice.append(dice_coef_np(imgs_mask_test[i,0], imgs_mask_pred[i,0]))
    print 'mean dice: ', np.mean(dice)


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        train_and_predict(False)
