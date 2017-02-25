import os
import sys
import numpy as np

import util
import image_aug
import luna_util
import luna_preprocess
from glob import glob


_SEED = 123456789
_HEIGHT = 512
_WIDTH = 512
_NUM_RAND_SLICES = 5
_NUM_AUG = 1
_OUTPUT_DIR = '../LUNA16/output_unet_data1'


def _make_aug(seed):
    return image_aug.ImageAug(
        rotate_angle_range=(-10, 10),
        zoom_factor_range=(0.8, 1.2),
        elastic_alpha_sigma=(50, 10),
        random_state=np.random.RandomState(seed))


def _transform(image, mask):
    try:
        bbox = util.find_bbox(image, margin=[2,2], bg=image[0,0])
        image = image[bbox]
        mask = mask[bbox]
    except:
        pass

    image = util.pad_to_square(image)
    mask = util.pad_to_square(mask)

    scale_factor = np.asarray([_HEIGHT, _WIDTH], dtype=np.float32) / image.shape

    image = image_aug.resize(image, [_HEIGHT, _WIDTH])
    mask = image_aug.resize(mask, [_HEIGHT, _WIDTH])    
    return image, mask, scale_factor


def _sample(image, mask, lung_aug=None, mask_aug=None):
    pos_ans = []
    neg_ans = []

    if lung_aug and mask_aug:
        image = lung_aug.apply(image)
        mask = mask_aug.apply(mask)

    image, mask, scale_factor = _transform(image, mask)
    #if np.min(scale_factor) > 5:
    #    return pos_ans, neg_ans

    if util.is_pos_mask(mask):
        pos_ans.append((image, mask, scale_factor))
    else:
        neg_ans.append((image, mask, scale_factor))
    return pos_ans, neg_ans


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
            for z_offset in [-1, 0, 1]:
                slice_zs.append(util.clip_dim0(masked_lung, nod_v_z + z_offset))
        # random sample slices
        for _ in range(_NUM_RAND_SLICES):
            slice_zs.append(random_state.randint(len(masked_lung)))
        slice_zs = list(set(slice_zs))

        for slice_z in slice_zs:
            new_image, new_nodule_mask = luna_util.slice_image(
                masked_lung, all_nodule_mask, slice_z)
            # Original
            pos_ans, neg_ans = _sample(
                new_image, new_nodule_mask, None, None)
            pos_out.extend(pos_ans)
            neg_out.extend(neg_ans)
            # Aug
            for _ in range(_NUM_AUG):
                pos_ans, neg_ans = _sample(
                    new_image, new_nodule_mask, lung_aug, mask_aug)
                pos_out.extend(pos_ans)
                neg_out.extend(neg_ans)

    all_out = luna_util.shuffle_out(pos_out, neg_out, random_state)
    if len(all_out) == 0:
        print 'Warning: skip %s'%data_dir
        return

    out_images = []
    out_nodule_masks = []
    out_scale_factors = []
    for t_image, t_nodule_mask, t_scale_factor in all_out:
        out_images.append(np.expand_dims(t_image, 0))
        out_nodule_masks.append(np.expand_dims(t_nodule_mask, 0))
        out_scale_factors.append(t_scale_factor)
    out_images = np.stack(out_images)
    out_nodule_masks = np.stack(out_nodule_masks)
    out_scale_factors = np.stack(out_scale_factors)

    np.savez_compressed(
        os.path.join(_OUTPUT_DIR, os.path.basename(data_dir)),
        images=out_images,
        nodule_masks=out_nodule_masks,
        scale_factors=out_scale_factors)    


def load_data(subsets):
    images = []
    nodule_masks = []
    scale_factors = []
    for subset in subsets:
        data = np.load(os.path.join(_OUTPUT_DIR, '%s.npz'%subset))
        images.append(data['images'].astype(np.float32))
        nodule_masks.append(data['nodule_masks'].astype(np.float32))
        scale_factors.append(data['scale_factors'].astype(np.float32))
    images = np.concatenate(images)
    nodule_masks = np.concatenate(nodule_masks)
    scale_factors = np.concatenate(scale_factors)
    return images, nodule_masks, scale_factors


if __name__ == '__main__':
    data_dirs = sorted(glob(luna_preprocess._DATA_DIR))
    for data_dir in data_dirs:
        process_data_dir(data_dir)
    print 'Done'
