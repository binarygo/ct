import os
import sys
import numpy as np
from skimage import transform
from glob import glob

import util
import image_aug
import luna_cropper
import luna_preprocess


_SEED = 12345
_CROP_HEIGHT = 96
_CROP_WIDTH = 96
_NUM_PATCHES = 5
_NUM_AUG = 5
_OUTPUT_DIR = '../LUNA16/output_unet_data2'


def _make_aug(seed):
    return image_aug.ImageAug(
        rotate_angle_range=(-10, 10),
        zoom_factor_range=(0.8, 1.2),
        elastic_alpha_sigma=(50, 10),
        random_state=np.random.RandomState(seed))


def _sample_patches_impl(image, mask, image_aug=None, mask_aug=None):
    if image_aug:
        image = image_aug.apply(image)

    if mask_aug:
        mask = mask_aug.apply(mask)
        
    pos_ans = []
    neg_ans = []
    c = luna_cropper.Cropper(image, mask, [_CROP_HEIGHT, _CROP_WIDTH])
    for i in range(_NUM_PATCHES):
        ans = c.crop_pos()
        if ans is not None:
            pos_ans.append(ans)
        ans = c.crop_neg()
        if ans is not None:
            neg_ans.append(ans)
    return pos_ans, neg_ans


def _sample_patches(image, mask, image_aug, mask_aug):
    pos_ans, neg_ans = _sample_patches_impl(image, mask, None, None)
    for i in range(_NUM_AUG):
        aug_pos_ans, aug_neg_ans = _sample_patches_impl(
            image, mask, image_aug, mask_aug)
        pos_ans += aug_pos_ans
        neg_ans += aug_neg_ans
    n = min(len(pos_ans), len(neg_ans))
    return pos_ans[:n], neg_ans[:n]


def process_data_dir(data_dir):
    random_state = np.random.RandomState(_SEED)
    image_aug = _make_aug(_SEED)
    mask_aug = _make_aug(_SEED)

    out_images = []
    out_nodule_masks = []
    file_list = glob(os.path.join(data_dir, '*.mhd'))
    for i, f in enumerate(file_list):
        f_id = luna_preprocess.get_file_id(f)
        print '========== process %s:%s of %d'%(data_dir, f_id, len(file_list))
        sys.stdout.flush()

        image = luna_preprocess.Image()
        try:
            image.load(f_id)
        except Exception as e:
            print 'Error: %s'%e
            continue

        masked_lung = image.masked_lung
        nodule_masks = image._nodule_masks

        nodules = image.get_v_nodules()
        for nod_idx in range(len(nodules)):
            nod_v_x, nod_v_y, nod_v_z, nod_v_d = nodules[nod_idx]
            nodule_mask = nodule_masks[nod_idx]
            for z_offset in [-1, 0, 1]:
                slice_z = np.clip(nod_v_z + z_offset, 0, masked_lung.shape[0] - 1)
                new_image = masked_lung[slice_z]
                new_image = util.normalize(new_image, 0.0)
                new_nodule_mask = nodule_mask[slice_z]
                pos_ans, neg_ans = _sample_patches(
                    new_image, new_nodule_mask, image_aug, mask_aug)
                for t_image, t_nodule_mask in pos_ans:
                    out_images.append(t_image)
                    out_nodule_masks.append(t_nodule_mask)
                for t_image, t_nodule_mask in neg_ans:
                    out_images.append(t_image)
                    out_nodule_masks.append(t_nodule_mask)                    

    out_len = len(out_images)
    if out_len == 0 or out_len != len(out_nodule_masks):
        print 'Warning: skip %s'%data_dir
        return

    perm_idxes = random_state.permutation(out_len)
    np.savez_compressed(
        os.path.join(_OUTPUT_DIR, os.path.basename(data_dir)),
        images=[out_images[i] for i in perm_idxes],
        nodule_masks=[out_nodule_masks[i] for i in perm_idxes])


def load_data(subsets):
    images = []
    nodule_masks = []
    for subset in subsets:
        data = np.load(os.path.join(_OUTPUT_DIR, '%s.npz'%subset))
        images.extend([x.astype(np.float32) for x in data['images']])
        nodule_masks.extend([x.astype(np.float32) for x in data['nodule_masks']])
    return images, nodule_masks


if __name__ == '__main__':
    data_dirs = sorted(glob(luna_preprocess._DATA_DIR))
    for data_dir in data_dirs:
        process_data_dir(data_dir)
    print 'Done'
