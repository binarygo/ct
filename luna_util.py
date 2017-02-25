import os
import numpy as np

import util


def slice_image(masked_lung, nodule_mask, slice_z):
    new_image = masked_lung[slice_z]
    new_image = util.normalize(new_image, 0.0)

    new_nodule_mask = None
    if nodule_mask is not None:
        new_nodule_mask = nodule_mask[slice_z]

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
