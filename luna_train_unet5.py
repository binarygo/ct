import os
import time

import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import image_aug
import luna_util
import luna_train_util
import luna_unet_data5


_SEED = 123456789
_DATA_DIR = '../LUNA16/output_unet_data5'
_MODEL_PATH = './unet5.hdf5'

_IMAGE_ROWS = 96
_IMAGE_COLS = 96

_EX_PERCENT = 0.75
_BATCH_SIZE = 16
_NUM_EPOCHS = 1000


def get_unet():
    model = luna_train_util.UnetModel(
        depths=[32, 64, 128, 256],
        poolings=True,
        inputs=Input((1, _IMAGE_ROWS, _IMAGE_COLS)),
        kernel_nb_row=3,
        kernel_nb_col=3,
        batch_norm_inputs=True,
        batch_norm=True,
        dropout_prob=0.5).make_model()

    model.compile(optimizer=Adam(lr=1.0e-5),
                  loss=luna_train_util.dice_coef_loss,
                  metrics=[luna_train_util.dice_coef])
    return model
        

def load_data(subsets, ex_percent=0):
    random_states = [
        np.random.RandomState(_SEED),
        np.random.RandomState(_SEED)
    ]
    def load_data_impl(label):
        return luna_util.shuffle_together(
            *luna_unet_data5.load_data([label]),
            random_states=random_states)
    images = []
    masks = []
    for subset in subsets:
        pos_label = 'pos_' + subset
        neg_label = 'neg_' + subset
        exneg_label = 'exneg_' + subset
        pos_images, pos_masks = load_data_impl(pos_label)
        if ex_percent <= 0:
            tneg_images, tneg_masks = load_data_impl(neg_label)
        elif ex_percent >= 1:
            tneg_images, tneg_masks = load_data_impl(exneg_label)
        else:
            neg_images, neg_masks = load_data_impl(neg_label)
            exneg_images, exneg_masks = load_data_impl(exneg_label)
            tneg_size = int(min(len(exneg_images) / ex_percent,
                                len(neg_images) / (1.0 - ex_percent)))
            exneg_size = int(tneg_size * ex_percent)
            neg_size = int(tneg_size * (1.0 - ex_percent))
            tneg_size = neg_size + exneg_size
            tneg_images = np.concatenate([
                neg_images[0:neg_size], exneg_images[0:exneg_size]])
            tneg_masks = np.concatenate([
                neg_masks[0:neg_size], exneg_masks[0:exneg_size]])
            tneg_imags, tneg_masks = luna_util.shuffle_together(
                tneg_images, tneg_masks, random_states)
        pos_size = len(pos_images)
        tneg_size = len(tneg_images)
        size = min(pos_size, tneg_size)
        images.extend([pos_images[0:size], tneg_images[0:size]])
        masks.extend([pos_masks[0:size], tneg_masks[0:size]])
    images = np.concatenate(images)
    masks = np.concatenate(masks)
    images, masks = luna_util.shuffle_together(
        images, masks, random_states)
    return images, masks
    

def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    subsets_train = []
    for i in [0]: #[0,1,2,3,4,5,6,7,8]:
        subsets_train.append('subset%d'%i)

    subsets_test = []
    for i in [9]:
        subsets_test.append('subset%d'%i)

    imgs_train, imgs_mask_train = load_data(
        subsets_train, ex_percent=_EX_PERCENT)
    imgs_test, imgs_mask_test = load_data(
        subsets_test, ex_percent=_EX_PERCENT)

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


def pred_nodule_mask(image, model):
    crop_h, crop_w = _IMAGE_ROWS//2, _IMAGE_COLS//2
    pad_h, pad_w = (_IMAGE_ROWS - crop_h)//2, (_IMAGE_COLS - crop_w)//2
    ans = np.zeros_like(image, dtype=np.float)
    nrows, ncols = np.ceil((np.asarray(image.shape) / [crop_h, crop_w])).astype(np.int)
    for i in range(nrows):
        for j in range(ncols):
            row_slice = slice(i * crop_h, (i + 1) * crop_h)
            col_slice = slice(j * crop_w, (j + 1) * crop_w)
            crop_yx = [crop_h * i - pad_h, crop_w * j - pad_w]
            image_patch = image_aug.crop(image, crop_yx, [_IMAGE_ROWS, _IMAGE_COLS])
            mask_patch = model.predict(
                np.reshape(image_patch, [1, 1, _IMAGE_ROWS, _IMAGE_COLS]))[0,0]
            ans[row_slice, col_slice] += mask_patch[pad_h:pad_h+crop_h, pad_w:pad_w+crop_w]
    return ans


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        train_and_predict(True)
