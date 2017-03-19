import os

import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import luna_train_util
import luna_unet_data5


_DATA_DIR = '../LUNA16/output_unet_data5'
_MODEL_PATH = './unet5.hdf5'

_IMAGE_ROWS = 128
_IMAGE_COLS = 128

_BATCH_SIZE = 32
_NUM_EPOCHS = 100


def get_unet():
    model = luna_train_util.make_unet(
        depths=[32, 64, 128, 256, 512],
        inputs=Input((1, _IMAGE_ROWS, _IMAGE_COLS)),
        kernel_nb_row=3,
        kernel_nb_col=3,
        batch_norm=True, dropout_prob=0.5)

    model.compile(optimizer=Adam(lr=1.0e-5),
                  loss=luna_train_util.dice_coef_loss,
                  metrics=[luna_train_util.dice_coef])
    return model
        

def load_data(subsets):
    images, nodule_masks = luna_unet_data5.load_data(subsets)
    return images, nodule_masks
    

def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    subsets_train = []
    for i in [0,1,2,3,4,5,6,7,8]:
        subsets_train.extend(['pos_subset%d'%i, 'neg_subset%d'%i])

    subsets_test = []
    for i in [9]:
        subsets_test.extend(['pos_subset%d'%i, 'neg_subset%d'%i])

    imgs_train, imgs_mask_train = load_data(subsets_train)
    imgs_test, imgs_mask_test = load_data(subsets_test)

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


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        train_and_predict(False)
