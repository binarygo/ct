import os
import sys
import numpy as np
from skimage import transform
from glob import glob

import util
import image_aug
import luna_util
import luna_cropper
import luna_preprocess


_SEED = 123456789
_CROP_HEIGHT = 128
_CROP_WIDTH = 128
_NUM_PATCHES = 15
_NUM_RAND_SLICES = 5
_NUM_AUGS = 2
_OUTPUT_DIR = '../LUNA16/output_unet_data5'


def _make_aug(seed):
    return image_aug.ImageAug(
        rotate_angle_range=(-10, 10),
        zoom_factor_range=(0.8, 1.2),
        elastic_alpha_sigma=(50, 10),
        random_state=np.random.RandomState(seed))


def _sample_patches(image, mask, lung_aug=None, mask_aug=None):
    if lung_aug and mask_aug:
        image = lung_aug.apply(image)
        mask = mask_aug.apply(mask)
        mask = util.to_bool_mask(mask).astype(np.float32)
        
    c = luna_cropper.Cropper(image, mask, [_CROP_HEIGHT, _CROP_WIDTH])
    return luna_util.crop_patches(c, _NUM_PATCHES, cpad_factor=0.5)


def process_data_dir(data_dir):
    random_state = np.random.RandomState(_SEED // 7)
    lung_aug = _make_aug(_SEED)
    mask_aug = _make_aug(_SEED)

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

        slice_zs = []
        # sample nodule slices
        nodules = image.get_v_nodules()
        for nod_idx in range(len(nodules)):
            nod_v_x, nod_v_y, nod_v_z, nod_v_d = nodules[nod_idx]
            for z_offset in [-2, -1, 0, 1, 2]:
                slice_zs.append(util.clip_dim0(masked_lung, nod_v_z + z_offset))
        # random sample slices
        for _ in range(_NUM_RAND_SLICES):
            slice_zs.append(random_state.randint(len(masked_lung)))
        slice_zs = list(set(slice_zs))

        tmp_pos_out = []
        tmp_neg_out = []
        for slice_z in slice_zs:
            new_image, new_nodule_mask = luna_util.slice_image(
                masked_lung, all_nodule_mask, slice_z)
            pos_ans, neg_ans = _sample_patches(
                new_image, new_nodule_mask, None, None)
            tmp_pos_out.extend(pos_ans)
            tmp_neg_out.extend(neg_ans)
            for _ in range(_NUM_AUGS):
                pos_ans, neg_ans = _sample_patches(
                    new_image, new_nodule_mask, lung_aug, mask_aug)
                tmp_pos_out.extend(pos_ans)
                tmp_neg_out.extend(neg_ans)
        tmp_neg_out = util.shuffle(tmp_neg_out, len(tmp_pos_out) * 2,
                                   random_state)
        pos_out.extend(tmp_pos_out)
        neg_out.extend(tmp_neg_out)

    print '# pos_out = %d'%len(pos_out)
    print '# neg_out = %d'%len(neg_out)
    out_len = min(len(pos_out), len(neg_out))
    if out_len == 0:
        print 'Warning: skip %s'%data_dir
        return

    pos_out = util.shuffle(pos_out, out_len, random_state)
    neg_out = util.shuffle(neg_out, out_len, random_state)
    for label, x_out in [('pos', pos_out), ('neg', neg_out)]:
        out_images = []
        out_nodule_masks = []
        for t_image, t_nodule_mask in x_out:
            out_images.append(np.expand_dims(t_image, 0))
            out_nodule_masks.append(np.expand_dims(t_nodule_mask, 0))
        out_images = np.stack(out_images)
        out_nodule_masks = np.stack(out_nodule_masks)

        np.savez_compressed(
            os.path.join(_OUTPUT_DIR, label + '_' + os.path.basename(data_dir)),
            images=out_images, nodule_masks=out_nodule_masks)    


def load_data(subsets):
    return luna_util.load_data(
        subsets, _OUTPUT_DIR, ['images', 'nodule_masks'])


def get_images_mean_and_std(subsets):
    return luna_util.get_mean_and_std(subsets, _OUTPUT_DIR, 'images');


if __name__ == '__main__':
    data_dirs = sorted(glob(luna_preprocess._DATA_DIR))
    for data_dir in data_dirs:
        print '========== process %s'%data_dir
        process_data_dir(data_dir)
    print 'Done'
