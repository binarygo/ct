import numpy as np

from keras.models import Model
from keras.layers import merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K

_DICE_SMOOTH = 1.

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2. * intersection + _DICE_SMOOTH) /
            (K.sum(y_true_f) + K.sum(y_pred_f) + _DICE_SMOOTH))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return ((2. * intersection + _DICE_SMOOTH) /
            (np.sum(y_true_f) + np.sum(y_pred_f) + _DICE_SMOOTH))


def _append(s, postfix):
    return s + postfix if s is not None else None


def _conv(nb_filter, nb_row, nb_col, input,
          batch_norm, dropout_prob,
          use_shortcut=False, name=None):
    ans = Convolution2D(nb_filter, nb_row, nb_col,
                        border_mode='same')(input)
    x = ans
    if batch_norm:
        ans = BatchNormalization(mode=1)(ans)
    ans = Activation('relu', name=name)(ans)
    if dropout_prob is not None:
        ans = Dropout(dropout_prob)(ans)
    if use_shortcut:
        ans = Convolution2D(nb_filter, nb_row, nb_col,
                            border_mode='same')(ans)
        if batch_norm:
            ans = BatchNormalization(mode=1)(ans)
        ans = merge([x, ans], mode='sum')
        ans = Activation('relu', name=_append(name,'_shortcut'))(ans)
        if dropout_prob is not None:
            ans = Dropout(dropout_prob)(ans)
    return ans


class UnetModel(object):

    def __init__(self,
                 depths,
                 poolings,
                 inputs,
                 kernel_nb_row,
                 kernel_nb_col,
                 batch_norm_inputs=False,
                 batch_norm=False,
                 dropout_prob=None,
                 use_shortcut=False):
        num_depths = len(depths)
        assert num_depths >= 2

        if isinstance(poolings, bool):
            poolings = [poolings] * (num_depths-1)
        poolings.append(None)
        assert num_depths == len(poolings)

        top = inputs
        if batch_norm_inputs:
            top = BatchNormalization(mode=1)(inputs)

        def conv_impl(nb_filter, input, name=None):
            return _conv(nb_filter, kernel_nb_row, kernel_nb_col,
                         input, batch_norm, dropout_prob,
                         use_shortcut=use_shortcut, name=name)

        convs = []
        for i in range(num_depths-1):
            depth = depths[i]
            conv = conv_impl(depth, top)
            conv = conv_impl(depth, conv)
            top = conv
            convs.append(conv)
            if poolings[i]:
                pool = MaxPooling2D(pool_size=(2, 2))(conv)
                top = pool

        conv = conv_impl(depths[-1], top)
        conv = conv_impl(depths[-1], conv, name='utip')
        mid = conv
        convs.append(conv)

        for i in range(num_depths-1):
            depth = depths[-i-2]
            pooling = poolings[-i-2]
            up = convs[num_depths-1+i]
            if pooling:
                up = UpSampling2D(size=(2, 2))(up)
            up = merge([up, convs[num_depths-2-i]],
                        mode='concat', concat_axis=1)
            conv = conv_impl(depth, up)
            conv = conv_impl(depth, conv)
            convs.append(conv)
            top = conv

        top = Convolution2D(1, 1, 1, activation='sigmoid')(top)

        self._inputs = inputs
        self._mid = mid
        self._outputs = top

    def make_model(self):
        return Model(input=self._inputs, output=self._outputs)
