import os
import sys
import numpy as np
from skimage import transform
from glob import glob

import tensorflow as tf

import util
import image_aug
import luna_util
import luna_cropper
import luna_preprocess
import luna_train_util

import luna_unet_data5
import luna_train_unet5


_SEED = 123456789
_MODEL_PATH = './save/unet5.hdf5-gpu-epoch-61-0p72'

_NUM_RAND_SLICES = 50
_DICE_THRESHOLD = 0.1

def process_data_dir(data_dir, model):
    random_state = np.random.RandomState(_SEED // 7)
    lung_aug = luna_unet_data5._make_aug(_SEED)
    mask_aug = luna_unet_data5._make_aug(_SEED)

    pos_out = []
    neg_out = []
    file_list = glob(os.path.join(data_dir, '*.mhd'))
    for i, f in enumerate(file_list):
        f_id = luna_preprocess.get_file_id(f)
        print '========== process %s:%s: %d of %d'%(data_dir, f_id, i + 1, len(file_list))
        sys.stdout.flush()

        image = luna_preprocess.Image()
        try:
            image.load(f_id)
        except Exception as e:
            print 'Error: %s'%e
            continue

        masked_lung = image.masked_lung
        all_nodule_mask = image._all_nodule_mask.astype(np.float32)

        # sample nodule slices
        nod_slice_zs = []
        nodules = image.get_v_nodules()
        for nod_idx in range(len(nodules)):
            nod_v_x, nod_v_y, nod_v_z, nod_v_d = nodules[nod_idx]
            nod_slice_zs.append(util.clip_dim0(masked_lung, nod_v_z))
        # random sample slices
        slice_zs = []
        for _ in range(_NUM_RAND_SLICES):
            slice_zs.append(random_state.randint(len(masked_lung)))
        slice_zs = list(set(slice_zs) - set(nod_slice_zs))

        for slice_z in slice_zs:
            new_image, new_nodule_mask = luna_util.slice_image(
                masked_lung, all_nodule_mask, slice_z)
            if random_state.randint(2):  # 50% prob
                pos_ans, neg_ans = luna_unet_data5._sample_patches(
                    new_image, new_nodule_mask, None, None)
            else:
                pos_ans, neg_ans = luna_unet_data5._sample_patches(
                    new_image, new_nodule_mask, lung_aug, mask_aug)
            for a_ans in neg_ans:
                image_patch, mask_patch = a_ans
                p_mask_patch = model.predict(
                    np.reshape(image_patch, [1, 1] + list(image_patch.shape)))[0,0]
                if luna_train_util.dice_coef_np(mask_patch, p_mask_patch) < _DICE_THRESHOLD:
                    neg_out.append(a_ans)

    print '# neg_out = %d'%len(neg_out)
    out_len = len(neg_out)
    if out_len == 0:
        print 'Warning: skip %s'%data_dir
        return

    neg_out = util.shuffle(neg_out, out_len, random_state)
    out_images = []
    out_nodule_masks = []
    for t_image, t_nodule_mask in neg_out:
        out_images.append(np.expand_dims(t_image, 0))
        out_nodule_masks.append(np.expand_dims(t_nodule_mask, 0))
    out_images = np.stack(out_images)
    out_nodule_masks = np.stack(out_nodule_masks)

    np.savez_compressed(
        os.path.join(luna_unet_data5._OUTPUT_DIR, 'exneg_' + os.path.basename(data_dir)),
        images=out_images, nodule_masks=out_nodule_masks)    


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        model = luna_train_unet5.get_unet()
    model.load_weights(_MODEL_PATH)

    data_dirs = sorted(glob(luna_preprocess._DATA_DIR))
    for data_dir in data_dirs:
        print '========== process %s'%data_dir
        process_data_dir(data_dir, model)
    print 'Done'
