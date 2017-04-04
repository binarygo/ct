import numpy as np
from skimage import transform
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

import util


def crop(image, crop_x, crop_size, bg=None):
    zero_x = tuple([0]*image.ndim)
    if bg is None:
        bg = image[zero_x]
    image_size = np.asarray(image.shape, dtype=np.int)
    crop_size = np.asarray(crop_size, dtype=np.int)
    crop_x0 = np.asarray(crop_x, dtype=np.int)
    crop_x1 = crop_x0 + crop_size

    t_x0 = np.minimum(np.maximum(zero_x, crop_x0), image_size)
    t_x1 = np.minimum(np.maximum(zero_x, crop_x1), image_size)

    pad_before = np.maximum(t_x0-crop_x0, 0)
    pad_after = np.maximum(crop_size-(pad_before+t_x1-t_x0), 0)

    slices = tuple([
        slice(t_x0[i], t_x1[i])
        for i in range(image.ndim)
    ])
    pads = tuple([
        (pad_before[i], pad_after[i])
        for i in range(image.ndim)
    ])

    return np.pad(image[slices], pads, mode='constant', constant_values=bg)

    
def rotate(image, angle):
    assert len(image.shape) == 2

    bg = image[0, 0]
    return ndimage.rotate(image, angle=angle, order=0,
                          mode='constant', cval=bg, reshape=True)


def zoom(image, factor):
    assert len(image.shape) == 2

    bg = image[0, 0]
    h, w = image.shape

    return ndimage.interpolation.zoom(
        image, zoom=[factor, factor], order=0, mode='constant', cval=bg)


def resize(image, new_size):
    assert len(image.shape) == 2

    bg = image[0, 0]
    return transform.resize(image, new_size, order=0, mode='constant', cval=bg)


def rand_elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    bg = image[0, 0]

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=bg) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=bg) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return ndimage.interpolation.map_coordinates(image, indices, order=1).reshape(shape)


class ImageAug(object):

    _ROTATE = 0
    _ZOOM = 1
    _ELASTIC = 2

    def __init__(self,
                 rotate_angle_range=(-10, 10),
                 zoom_factor_range=(0.8, 1.2),
                 elastic_alpha_sigma=(50, 10),
                 random_state=None):
        self._rotate_angle_range = rotate_angle_range
        self._zoom_factor_range = zoom_factor_range
        self._elastic_alpha_sigma = elastic_alpha_sigma
        if random_state is None:
            random_state = np.random.RandomState(None)
        self._random_state = random_state

    def _rand(self, a, b):
        # [a, b)
        return a + self._random_state.rand() * (b - a)

    def apply(self, image, debug=False):
        assert len(image.shape) == 2

        seq = self._random_state.permutation([self._ROTATE, self._ZOOM, self._ELASTIC])
        seq_mask = int(self._rand(1, 2**len(seq)))
        seq_mask = [bool(int(x)) for x in bin(seq_mask)[2:]]

        ans = image
        trace = [ans]
        for s, s_on in zip(seq, seq_mask):
            if not s_on:
                continue
            if s == self._ROTATE:
                rotate_angle = self._rand(*self._rotate_angle_range)
                ans = rotate(ans, rotate_angle)
                trace.append((s, rotate_angle, ans))
            elif s == self._ZOOM:
                zoom_factor = self._rand(*self._zoom_factor_range)
                ans = zoom(ans, zoom_factor)
                trace.append((s, zoom_factor, ans))
            elif s == self._ELASTIC:
                ans = rand_elastic_transform(
                    ans,
                    alpha=self._elastic_alpha_sigma[0],
                    sigma=self._elastic_alpha_sigma[1],
                    random_state=self._random_state)
                trace.append(ans)
        if debug:
            return ans, trace
        return ans
