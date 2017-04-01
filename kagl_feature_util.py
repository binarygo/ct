import numpy as np
from scipy import stats
from skimage import measure

import util
import kagl_preprocess
import kagl_test_unet5


class Patient(object):

    def __init__(self, stage, name):
        image = kagl_preprocess.Image(stage)
        image.load(name)
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
        self._masked_lung = masked_lung
        self._nodule_mask = nodule_mask
        self._nodule_labels = nodule_labels
        self._nodule_regions = nodule_regions
        self._nodule_bboxes = nodule_bboxes

    @property
    def num_nodules(self):
        return len(self._nodule_regions)

    def nodule_local(self, x, nodule_idx):
        # x.shape == self._image.shape
        return x[self._nodule_bboxes[nodule_idx]]

    def nodule_local_image(self, nodule_idx):
        return self.nodule_local(self._masked_lung, nodule_idx)

    def nodule_local_mask(self, nodule_idx):
        return self.nodule_local(self._nodule_mask, nodule_idx)

    def nodule_local_masked_image(self, nodule_idx):
        return util.apply_mask(self.nodule_local_image(nodule_idx),
                               self.nodule_local_mask(nodule_idx))


def extract_array_features(a):
    a = a.flatten()
    return [
        len(a),
        np.mean(a),
        np.mean(np.abs(a)),
        np.std(a),
        stats.skew(a),
        stats.kurtosis(a)
    ] + list(np.percentile(a, [0, 25, 50, 75, 100]))


def logloss(act, pred):
    act = np.asarray(act, dtype=float)
    pred = np.asarray(pred, dtype=float)
    eps = 1e-15
    pred = np.clip(pred, eps, 1.0-eps)
    ll = -np.mean(act * np.log(pred) +
                  (1.0 - act) * np.log(1.0 - pred))
    return ll


def write_submission_file(patient_names, preds, file_name=''):
    with open('lcad-{}.csv'.format(file_name), 'w') as f:
        f.write('id,cancer\n')
        for i in range(len(patient_names)):
            f.write('{},{}\n'.format(patient_names[i], preds[i]))
