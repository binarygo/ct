import os
import time
import numpy as np

import util
import luna_cropper


def slice_image(masked_lung, nodule_mask, slice_z):
    new_image = masked_lung[slice_z]
    new_image = util.normalize(new_image, 0.0)

    new_nodule_mask = None
    if nodule_mask is not None:
        new_nodule_mask = nodule_mask[slice_z]

    return new_image, new_nodule_mask


def batch_slice_image(masked_lung, nodule_mask, slice_zs):
    new_image = np.stack([masked_lung[z] for z in slice_zs])
    new_image = util.normalize(new_image, 0.0)

    new_nodule_mask = None
    if nodule_mask is not None:
        new_nodule_mask = np.stack([nodule_mask[z] for z in slice_zs])

    return new_image, new_nodule_mask


def shuffle_out(pos_out, neg_out, random_state):
    print '# pos = %d'%len(pos_out)
    print '# neg = %d'%len(neg_out)

    out_len = min(len(pos_out), len(neg_out))
    if out_len == 0:
        return []

    pos_out = util.shuffle(pos_out, out_len, random_state)
    neg_out = util.shuffle(neg_out, out_len, random_state)
    all_out = util.shuffle(pos_out + neg_out, None, random_state)
    return all_out


def crop_patches(cropper, num_patches, cpad_factor=1.0):
    pos_ans = []
    neg_ans = []
    crop_yxs = set()
    for i in range(num_patches):
        # pos
        ans = cropper.crop_pos(cpad_factor=cpad_factor)
        if ans is not None:
            image_patch, mask_patch, crop_yx = ans
            if crop_yx not in crop_yxs:
                pos_ans.append((image_patch, mask_patch))
                crop_yxs.add(crop_yx)
        # neg
        ans = cropper.crop_neg()
        if ans is not None:
            image_patch, mask_patch, crop_yx = ans
            if crop_yx not in crop_yxs:
                neg_ans.append((image_patch, mask_patch))
                crop_yxs.add(crop_yx)
    return pos_ans, neg_ans

def batch_crop_patches(cropper, num_patches, image, mask):
    num_z = len(image)
    assert num_z == len(mask)

    def crop_patch(crop_yx):
        image_patch = np.stack([cropper.apply_crop(image[z], crop_yx)
                                for z in range(num_z)])
        mask_patch = np.stack([cropper.apply_crop(mask[z], crop_yx)
                               for z in range(num_z)])
        return image_patch, mask_patch

    pos_ans = []
    neg_ans = []
    crop_yxs = set()
    for i in range(num_patches):
        # pos
        ans = cropper.crop_pos()
        if ans is not None:
            _, _, crop_yx = ans
            if crop_yx not in crop_yxs:
                pos_ans.append(crop_patch(crop_yx))
                crop_yxs.add(crop_yx)
        # neg
        ans = cropper.crop_neg()
        if ans is not None:
            _, _, crop_yx = ans
            if crop_yx not in crop_yxs:
                neg_ans.append(crop_patch(crop_yx))
                crop_yxs.add(crop_yx)
    return pos_ans, neg_ans


def load_data(subsets, output_dir, keys):
    ans = []
    for i in range(len(keys)):
        ans.append([])
    for subset in subsets:
        data = np.load(os.path.join(output_dir, '%s.npz'%subset))
        for i, k in enumerate(keys):
            ans[i].append(data[k].astype(np.float32))
    for i in range(len(ans)):
        ans[i] = np.concatenate(ans[i])
    return tuple(ans)


def get_mean_and_std(subsets, output_dir, key):
    acc_n = 0
    acc_mean = 0.0
    acc_var = 0.0
    for subset in subsets:
        images = load_data([subset], output_dir, [key])
        n = images.size
        acc_n += n
        acc_mean += np.mean(images) * n
        acc_var += np.var(images) * n
    return float(acc_mean) / acc_n, np.sqrt(float(acc_var) / acc_n)


def shuffle_together(images, masks, seed=None):
    if seed is None:
        seed = int(time.time())
    images = np.stack(util.shuffle(
        images, random_state=np.random.RandomState(seed)))
    masks = np.stack(util.shuffle(
        masks, random_state=np.random.RandomState(seed)))
    return images, masks
