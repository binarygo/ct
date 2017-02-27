import numpy as np
from skimage import measure

import util
import image_aug


class Cropper(object):

    def __init__(self, image, nodule_mask, crop_shape, random_state=None):
        self._image = image
        self._nodule_mask = nodule_mask
        self._h, self._w = image.shape
        self._ch, self._cw = crop_shape
        self._size = np.asarray([self._h, self._w], dtype=np.float)
        self._csize = np.asarray([self._ch, self._cw], dtype=np.float)
        nod_yxs = []
        nodule_mask_labels = measure.label(util.to_bool_mask(nodule_mask))
        for r in measure.regionprops(nodule_mask_labels):
            nod_yxs.append(np.asarray(r.centroid, dtype=np.float))
        self._nod_yxs = nod_yxs

        if random_state is None:
            random_state = np.random.RandomState(None)
        self._random_state = random_state
        
    def _crop_impl(self, hero, y, x):
        return image_aug.crop(hero, (y, x), [self._ch, self._cw])

    def _normalize_yx(self, yx):
        yx = np.minimum(np.maximum(yx, [0, 0]), [self._h-1, self._w-1])
        return yx.astype(np.int)

    def crop(self):
        yx_max = np.maximum([0, 0], self._size - self._csize)
        yx = yx_max * self._random_state.rand(2)
        y, x = self._normalize_yx(yx)
        return (self._crop_impl(self._image, y, x),
                self._crop_impl(self._nodule_mask, y, x),
                (y, x))
    
    def crop_neg(self):
        for i in range(1000):
            image, nodule_mask, yx = self.crop()
            if not util.is_pos_mask(nodule_mask):
                return image, nodule_mask, yx
    
    def crop_pos(self, nod_yx=None):
        if nod_yx is None:
            if not self._nod_yxs:
                return
            nod_idx = self._random_state.randint(len(self._nod_yxs))
            nod_yx = self._nod_yxs[nod_idx]
        yx_min = np.maximum([0, 0], nod_yx - self._csize)
        yx_max = np.maximum([0, 0], self._size - self._csize)
        yx_max = np.minimum(yx_max, nod_yx)
        yx_max = np.maximum(yx_max, yx_min)
        yx = yx_min + (yx_max - yx_min) * self._random_state.rand(2)
        y, x = self._normalize_yx(yx)
        image = self._crop_impl(self._image, y, x)
        nodule_mask = self._crop_impl(self._nodule_mask, y, x)
        if util.is_pos_mask(nodule_mask):
            return image, nodule_mask, (y,x)
