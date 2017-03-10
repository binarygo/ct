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
_CROP_HEIGHT = 96
_CROP_WIDTH = 96
_CROP_DEPTH = 3
_NUM_PATCHES = 10
_NUM_RAND_SLICES = 1
_OUTPUT_DIR = '../LUNA16/output_unet_data4'


def _make_aug(seed):
    return image_aug.ImageAug(
        rotate_angle_range=(-10, 10),
        zoom_factor_range=(0.8, 1.2),
        elastic_alpha_sigma=(50, 10),
        random_state=np.random.RandomState(seed))


def _sample_patches(image, mask, lung_aug=None, mask_aug=None):
    if lung_aug and mask_aug:
        image = np.stack([lung_aug[i].apply(image[i])
                          for i in range(_CROP_DEPTH)])
        mask = np.stack([mask_aug[i].apply(mask[i])
                         for i in range(_CROP_DEPTH)])
        
    c = luna_cropper.Cropper(image[_CROP_DEPTH//2], mask[_CROP_DEPTH//2],
                             [_CROP_HEIGHT, _CROP_WIDTH])
    return luna_util.batch_crop_patches(c, _NUM_PATCHES, image, mask)


def process_data_dir(data_dir):
    random_state = np.random.RandomState(_SEED // 7)
    lung_aug = [_make_aug(_SEED) for _ in range(_CROP_DEPTH)]
    mask_aug = [_make_aug(_SEED) for _ in range(_CROP_DEPTH)]

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
            slice_zs.append(util.clip_dim0(masked_lung, nod_v_z))
        # random sample slices
        for _ in range(_NUM_RAND_SLICES):
            slice_zs.append(random_state.randint(len(masked_lung)))
        slice_zs = list(set(slice_zs))

        for slice_z in slice_zs:
            zs = [util.clip_dim0(masked_lung, slice_z+i-_CROP_DEPTH//2)
                  for i in range(_CROP_DEPTH)]
            if len(set(zs)) != len(zs):
                continue
            new_image, new_nodule_mask = luna_util.batch_slice_image(
                masked_lung, all_nodule_mask, zs)
            if random_state.randint(2):  # 50% prob
                pos_ans, neg_ans = _sample_patches(
                    new_image, new_nodule_mask, None, None)
            else:
                pos_ans, neg_ans = _sample_patches(
                    new_image, new_nodule_mask, lung_aug, mask_aug)
            pos_out.extend(pos_ans)
            neg_out.extend(neg_ans)

    all_out = luna_util.shuffle_out(pos_out, neg_out, random_state)
    if len(all_out) == 0:
        print 'Warning: skip %s'%data_dir
        return

    out_images = []
    out_nodule_masks = []
    for t_image, t_nodule_mask in all_out:
        out_images.append(t_image)
        out_nodule_masks.append(
            t_nodule_mask[_CROP_DEPTH//2:_CROP_DEPTH//2+1])
    out_images = np.stack(out_images)
    out_nodule_masks = np.stack(out_nodule_masks)

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
        print '========== process %s'%data_dir
        process_data_dir(data_dir)
    print 'Done'
