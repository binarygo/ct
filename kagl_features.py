import sys
import numpy as np
import pandas as pd
from skimage import measure
from matplotlib import pyplot as plt

import util
import luna_preprocess
import kagl_preprocess
import kagl_test_unet5


def ball3d_volume(r):
    return 4.0/3.0*np.pi*(r**3)


def feature_columns():
    columns = ['area', 'mean', 'std']
    for a in ['z', 'y', 'x']:
        for f in ['area', 'eccentricity', 'equivalent_diameter',
                  'perimeter', 'solidity']:
            columns.append(a + '_' + f)
    return columns


def extract_region_features(region, masked_lung):
    local_mask = region.filled_image
    min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
    local_image = masked_lung[min_z:max_z, min_y:max_y, min_x:max_x]
    masked_local_image = util.apply_mask(local_image, local_mask)
    
    ans = [
        region.area,
        np.mean(masked_local_image),
        np.std(masked_local_image)
    ]
    for axis in range(3):
        view = np.rollaxis(local_mask, axis)
        vmask = view[np.argmax(np.sum(view, axis=(1,2)))]
        vregion = measure.regionprops(vmask.astype(int))[0]
        ans.extend([
            vregion.area,
            vregion.eccentricity,
            vregion.equivalent_diameter,
            vregion.perimeter,
            vregion.solidity
        ])
    return ans


def extract_patient_features(stage, patient_name):
    image = kagl_preprocess.Image(stage)
    image.load(patient_name)
    masked_lung = image.masked_lung
    nodule_mask = kagl_test_unet5.load_nodule_mask(patient_name, stage)
    nodule_mask = util.to_bool_mask(nodule_mask)
    nodule_mask_labels = measure.label(nodule_mask)
    regions = measure.regionprops(nodule_mask_labels)
    
    ans = []
    MIN_D, MAX_D = 3, 33
    for r in regions:
        if (r.area < ball3d_volume(MIN_D/2.0) or
            r.area > ball3d_volume(MAX_D/2.0) or
            any(np.array(r.filled_image.shape)<=1)):
            continue
        f = extract_region_features(r, masked_lung)
        ans.append(f)
    
    return ans


if __name__ == '__main__':
    stage = 'stage1'
    meta_patient = kagl_preprocess.MetaPatient(stage)
    ans = {}
    for name, label in meta_patient.labels.iteritems():
        print '========== Process %s: %d of %d'%(
            name, len(ans) + 1, len(meta_patient.labels))
        sys.stdout.flush()
        f = pd.DataFrame(extract_patient_features(stage, name),
                         columns=feature_columns())
        ans[name] = f
    np.save('kagl_output_features.npy', ans)
