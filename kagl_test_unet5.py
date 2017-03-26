import os
import sys
import time
import numpy as np
import tensorflow as tf

import util
import luna_util
import luna_train_unet5
import kagl_preprocess


_MODEL_PATH = './unet5.hdf5'


def get_output_dir(stage):
    return os.path.join(kagl_preprocess._DATA_DIR, stage + '_test_unet5')


def _get_model():
    model = luna_train_unet5.get_unet()
    model.load_weights(_MODEL_PATH)
    return model


def _get_all_patient_names(stage):
    meta_patient = kagl_preprocess.MetaPatient(stage)
    return meta_patient.labels.keys()


def _get_fnames(a_dir, postfix):
    return [
        f[0:-len(postfix)] for f in os.listdir(a_dir)
        if f.endswith(postfix)
    ]
    

def _get_preprocessed_patient_names(stage):
    return _get_fnames(kagl_preprocess.get_output_dir(stage),
                       '_digest.npz')


def _get_done_patient_names(stage):
    return _get_fnames(get_output_dir(stage),
                       '_mask.npz')
    

def _test_unet5(patient_name, model, stage):
    image = kagl_preprocess.Image(stage)
    image.load(name)
    lung_mask = image._lung_mask
    masked_lung = image.masked_lung
    nodule_mask = []
    for slice_z in range(len(masked_lung)):
        t_image, _ = luna_util.slice_image(masked_lung, None, slice_z)
        t_mask = luna_train_unet5.pred_nodule_mask(t_image, model)
        util.apply_mask_impl(t_mask, lung_mask[slice_z], 0)
        nodule_mask.append(t_mask)
    nodule_mask = np.stack(nodule_mask)
    np.savez_compressed(
        os.path.join(get_output_dir(stage), patient_name + '_mask.npz'),
        nodule_mask=nodule_mask)


def load_nodule_mask(patient_name, stage):
    ans = np.load(os.path.join(get_output_dir(stage), patient_name + '_mask.npz'))
    return ans['nodule_mask']


if __name__ == '__main__':
    stage = 'stage1'

    with tf.device('/gpu:0'):
        model = _get_model()

    all_names = set(_get_all_patient_names(stage))
    print '********** all: %d'%len(all_names)

    done_names = set(_get_done_patient_names(stage))&all_names
    print '********** done: %d'%len(done_names)

    while len(done_names) != len(all_names):
        names = set(_get_preprocessed_patient_names(stage))
        names = (names&all_names)-done_names
        print '********** tbd: %d'%len(names)
        for i, name in enumerate(names):
            print '========== process %s: %d of %d'%(name, i+1, len(names))
            _test_unet5(name, model, stage)
        print 'sleep ...'
        os.sleep(300)  # 5min
