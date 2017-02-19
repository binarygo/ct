import os
import sys
import numpy as np
from skimage import transform
from glob import glob

import util
import image_aug
import luna_cropper
import luna_preprocess


_SEED = 123456789
_CROP_HEIGHT = 96
_CROP_WIDTH = 96
_NUM_PATCHES = 10
_OUTPUT_DIR = '../LUNA16/output_unet_data2'


def _make_aug(seed):
    return image_aug.ImageAug(
        rotate_angle_range=(-10, 10),
        zoom_factor_range=(0.8, 1.2),
        elastic_alpha_sigma=(50, 10),
        random_state=np.random.RandomState(seed))


def _sample_patches(image, mask, image_aug=None, mask_aug=None):
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


def process_data_dir(data_dir):
    random_state = np.random.RandomState(_SEED // 7)
    image_aug = _make_aug(_SEED)
    mask_aug = _make_aug(_SEED)

    pos_out_images = []
    pos_out_nodule_masks = []
    neg_out_images = []
    neg_out_nodule_masks = []
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
        all_nodule_mask = image._all_nodule_mask

        nodules = image.get_v_nodules()
        for nod_idx in range(len(nodules)):
            nod_v_x, nod_v_y, nod_v_z, nod_v_d = nodules[nod_idx]
            for z_offset in [-1, 0, 1]:
                slice_z = np.clip(nod_v_z + z_offset, 0, masked_lung.shape[0] - 1)
                new_image = masked_lung[slice_z]
                new_image = util.normalize(new_image, 0.0)
                new_nodule_mask = all_nodule_mask[slice_z]
                if random_state.randint(2):
                    pos_ans, neg_ans = _sample_patches(
                        new_image, new_nodule_mask, None, None)
                else:
                    pos_ans, neg_ans = _sample_patches(
                        new_image, new_nodule_mask, image_aug, mask_aug)
                for t_image, t_nodule_mask in pos_ans:
                    pos_out_images.append(t_image)
                    pos_out_nodule_masks.append(t_nodule_mask)
                for t_image, t_nodule_mask in neg_ans:
                    neg_out_images.append(t_image)
                    neg_out_nodule_masks.append(t_nodule_mask)                    

    assert len(pos_out_images)==len(pos_out_nodule_masks)
    assert len(neg_out_images)==len(neg_out_nodule_masks)

    out_len = min(len(pos_out_images), len(neg_out_images))
    if out_len == 0:
        print 'Warning: skip %s'%data_dir
        return

    out_images = pos_out_images[0:out_len] + neg_out_images[0:out_len]
    out_nodule_masks = pos_out_nodule_masks[0:out_len] + neg_out_nodule_masks[0:out_len]

    perm_idxes = random_state.permutation(2*out_len)
    out_images = np.stack(
        [np.expand_dims(out_images[i], 0) for i in perm_idxes])
    out_nodule_masks = np.stack(
        [np.expand_dims(out_nodule_masks[i], 0) for i in perm_idxes])

    np.savez_compressed(
        os.path.join(_OUTPUT_DIR, os.path.basename(data_dir)),
        images=out_images, nodule_masks=out_nodule_masks)


def load_data(subsets):
    images = []
    nodule_masks = []
    for subset in subsets:
        data = np.load(os.path.join(_OUTPUT_DIR, '%s.npz'%subset))
        images.append(data['images'].astype(np.float32))
        nodule_masks.append(data['nodule_masks'].astype(np.float32))
    images = np.concatenate(images)
    nodule_masks = np.concatenate(nodule_masks)
    return images, nodule_masks


def get_images_mean_and_std(subsets):
    acc_n = 0
    acc_mean = 0.0
    acc_var = 0.0
    for subset in subsets:
        images, nodule_masks = load_data([subset])
        n = images.size
        acc_n += n
        acc_mean += np.mean(images) * n
        acc_var += np.var(images) * n
    return float(acc_mean) / acc_n, np.sqrt(float(acc_var) / acc_n)


if __name__ == '__main__':
    data_dirs = sorted(glob(luna_preprocess._DATA_DIR))
    for data_dir in data_dirs:
        process_data_dir(data_dir)
    print 'Done'
