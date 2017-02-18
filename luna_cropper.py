import numpy as np
from skimage import measure

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
        nodule_mask_labels = measure.label(nodule_mask)
        for r in measure.regionprops(nodule_mask_labels):
            nod_yxs.append(np.asarray(r.centroid, dtype=np.float))
        self._nod_yxs = nod_yxs

        if random_state is None:
            random_state = np.random.RandomState(None)
        self._random_state = random_state
        
    def _crop_impl(self, hero, y, x):
        return image_aug.crop(hero, (y, x), [self._ch, self._cw])

    def _normalize_yx(self, yx):
        yx = np.minimum(np.maximum(yx, [0, 0]), [self._h, self._w])
        return yx.astype(np.int)

    def crop(self):
        f = 0.2
        offset = self._random_state.rand(2)
        offset *= (self._size - (1.0 - 2.0 * f) * self._csize)
        yx = -f * self._size + offset
        y, x = self._normalize_yx(yx)
        return (self._crop_impl(self._image, y, x),
                self._crop_impl(self._nodule_mask, y, x))
    
    def crop_neg(self):
        for i in range(1000):
            image, nodule_mask = self.crop()
            if np.sum(np.abs(nodule_mask)) < 0.5:
                return image, nodule_mask
    
    def crop_pos(self):
        if not self._nod_yxs:
            return
        nod_yx_idx = self._random_state.randint(len(self._nod_yxs))
        nod_yx = self._nod_yxs[nod_yx_idx]
        offset = self._random_state.rand(2)
        offset *= self._csize
        yx = (nod_yx - offset)
        y, x = self._normalize_yx(yx)
        image = self._crop_impl(self._image, y, x)
        nodule_mask = self._crop_impl(self._nodule_mask, y, x)
        if np.sum(np.abs(nodule_mask)) >= 0.5:
            return image, nodule_mask
