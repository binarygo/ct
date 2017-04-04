import numpy as np
from scipy import stats
from skimage import measure

import util
import image_aug
import kagl_preprocess
import kagl_test_unet5


class LungGeometry(object):
    
    def __init__(self, lung_mask):
        z, y, x = lung_mask.nonzero()
        center_z, center_y, center_x = np.mean(z), np.mean(y), np.mean(x)
        
        dz = []
        r = []
        for z in range(len(lung_mask)):
            y, x = lung_mask[z].nonzero()
            if len(y) == 0 or len(x) == 0:
                continue
            dz.append(z - center_z)
            r.append(np.max(util.distance(x, y, center_x, center_y))) 

        min_dz, max_dz = np.min(dz), np.max(dz)
        min_r, max_r, mean_r, std_r = np.min(r), np.max(r), np.mean(r), np.std(r)

        self._lung_mask = lung_mask
        self._center_z = center_z
        self._center_y = center_y
        self._center_x = center_x

        self._min_dz = min_dz
        self._max_dz = max_dz

        self._min_r = min_r
        self._max_r = max_r
        self._mean_r = mean_r
        self._std_r = std_r

    def z_section(self, iz):
        return self._lung_mask[iz].nonzero()


class Patient(object):

    def __init__(self, stage, name):
        image = kagl_preprocess.Image(stage)
        image.load(name)
        lung_mask = image._lung_mask
        masked_lung = image.masked_lung
        nodule_mask = kagl_test_unet5.load_nodule_mask(name, stage)
        nodule_mask = util.to_bool_mask(nodule_mask)
        nodule_labels = measure.label(nodule_mask)
        nodule_regions = measure.regionprops(nodule_labels)
        nodule_bboxes = []
        for r in nodule_regions:
            min_z, min_y, min_x, max_z, max_y, max_x = r.bbox
            nodule_bboxes.append(np.s_[min_z:max_z, min_y:max_y, min_x:max_x])

        self._image = image
        self._lung_mask = lung_mask
        self._masked_lung = masked_lung
        self._lung_geometry = LungGeometry(lung_mask)

        self._nodule_mask = nodule_mask
        self._nodule_labels = nodule_labels
        self._nodule_regions = nodule_regions
        self._nodule_bboxes = nodule_bboxes

    @property
    def num_nodules(self):
        return len(self._nodule_regions)


    def nodule_position(self, nodule_idx):
        bbox = self._nodule_bboxes[nodule_idx]
        return [(s.start+s.stop)/2.0 for s in bbox]


    def nodule_local(self, t_image, nodule_idx, new_size=None):
        assert t_image.shape == self._lung_mask.shape
        bbox = self._nodule_bboxes[nodule_idx]
        if new_size is None:
            return t_image[bbox]
        x = [int((s.start+s.stop)/2.0-new_size[i]/2.0)
             for i, s in enumerate(bbox)]
        return image_aug.crop(t_image, x, new_size)

    def nodule_local_image(self, nodule_idx, new_size=None):
        return self.nodule_local(self._masked_lung, nodule_idx, new_size)

    def nodule_local_mask(self, nodule_idx, new_size=None):
        nodule_label = self._nodule_regions[nodule_idx].label
        return self.nodule_local(self._nodule_labels==nodule_label,
                                 nodule_idx, new_size)

    def nodule_local_masked_image(self, nodule_idx, new_size=None):
        return util.apply_mask(self.nodule_local_image(nodule_idx, new_size),
                               self.nodule_local_mask(nodule_idx, new_size))


def mask_diff(mask1, mask2):
    ans = mask1.copy()
    ans[mask2] = False
    return ans


def nodule_boundary_diff(patient, nodule_idx):
    p = patient
    nod_region = p._nodule_regions[nodule_idx]
    nod_r = util.ball3d_r(nod_region.area)
    band_r = max(1.0, 0.2 * nod_r)
    #band_volume = (util.ball3d_volume(nod_r + band_r / 2.0) -
    #               util.ball3d_volume(nod_r - band_r / 2.0))
    
    cage_size = [64]*3
    nod_image = p.nodule_local_image(nodule_idx, cage_size)
    nod_mask = p.nodule_local_mask(nodule_idx, cage_size)
    d_mask = util.dilate_mask_impl(nod_mask, band_r)
    e_mask = util.erode_mask_impl(nod_mask, band_r)
    band_volume = max(1, np.sum(d_mask) - np.sum(e_mask)) / 2.0
    
    out_mask = mask_diff(d_mask, nod_mask)
    in_mask = mask_diff(nod_mask, e_mask)
    
    out_val = np.mean(util.apply_mask(nod_image, out_mask))
    in_val = np.mean(util.apply_mask(nod_image, in_mask))
    
    return (in_val - out_val) / band_volume


def extract_array_features(a):
    a = a.flatten()
    return [
        np.mean(a),
        np.min(a),
        np.max(a),
        np.sum(a),
        np.std(a),
        stats.skew(a),
        stats.kurtosis(a)
    ]


def logloss(act, pred):
    act = np.asarray(act, dtype=float)
    pred = np.asarray(pred, dtype=float)
    eps = 1e-15
    pred = np.clip(pred, eps, 1.0-eps)
    ll = -np.mean(act * np.log(pred) +
                  (1.0 - act) * np.log(1.0 - pred))
    return ll


def write_submission_file(patient_names, preds, file_name=''):
    fname = 'lcad-{}.csv'.format(file_name)
    with open(fname, 'w') as f:
        f.write('id,cancer\n')
        for name, pred in sorted(zip(patient_names, preds)):
            f.write('{},{}\n'.format(name, pred))
    print 'Write {}'.format(fname)
    return fname
