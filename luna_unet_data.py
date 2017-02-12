import os
import sys
import numpy as np
from skimage import transform

import util
import luna_preprocess
from glob import glob


_OUTPUT_DIR = '../LUNA16/output_unet_data'


def process_data_dir(data_dir):
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
                new_image = transform.resize(
                    util.normalize(util.pad_to_square(new_image), 0.0),
                    [512, 512])
                new_nodule_mask = nodule_mask[slice_z]
                new_nodule_mask = transform.resize(
                    util.pad_to_square(new_nodule_mask),
                    [512, 512])
                out_images.append(np.expand_dims(new_image, 0))
                out_nodule_masks.append(np.expand_dims(new_nodule_mask, 0))

    if len(out_images) > 0 and len(out_nodule_masks) > 0:
        final_images = np.stack(out_images)
        final_nodule_masks = np.stack(out_nodule_masks)
        np.savez_compressed(
            os.path.join(_OUTPUT_DIR, os.path.basename(data_dir)),
            images=final_images,
            nodule_masks=final_nodule_masks)
    else:
        print 'Warning: skip %s'%data_dir


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


if __name__ == '__main__':
    data_dirs = sorted(glob(luna_preprocess._DATA_DIR))
    for data_dir in data_dirs:
        process_data_dir(data_dir)
    print 'Done'
