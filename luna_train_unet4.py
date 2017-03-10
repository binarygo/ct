import os

import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import luna_train_util
import luna_unet_data4


_DATA_DIR = '../LUNA16/output_unet_data4'
_MODEL_PATH = './unet4.hdf5'

_IMAGE_DEPTH = 3
_IMAGE_ROWS = 96
_IMAGE_COLS = 96

_IMAGES_MEAN = 0.180
_IMAGES_STD = 0.270

_BATCH_SIZE = 6
_NUM_EPOCHS = 100


def get_unet():
    model = luna_train_util.make_unet(
        depths=[32, 64, 128, 256, 512],
        inputs=Input((_IMAGE_DEPTH, _IMAGE_ROWS, _IMAGE_COLS)),
        kernel_nb_row=3,
        kernel_nb_col=3,
        batch_norm=True, dropout_prob=0.5)

    model.compile(optimizer=Adam(lr=1.0e-5),
                  loss=luna_train_util.dice_coef_loss,
                  metrics=[luna_train_util.dice_coef])
    return model
        

def normalize_images(images):
    return (images - _IMAGES_MEAN) / _IMAGES_STD


def load_data(subsets):
    images, nodule_masks = luna_unet_data4.load_data(subsets)
    images = normalize_images(images)
    return images, nodule_masks
    

def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

#    imgs_train, imgs_mask_train = load_data(
#        ['subset0', 'subset1', 'subset2',
#         'subset3', 'subset4', 'subset5',
#         'subset6', 'subset7', 'subset8'])
    imgs_train, imgs_mask_train = load_data(['subset0'])

    imgs_test, imgs_mask_test = load_data(['subset9'])

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # Saving weights to unet4.hdf5 at checkpoints
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
    with tf.device('/cpu:0'):
        train_and_predict(False)
