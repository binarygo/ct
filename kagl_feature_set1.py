import sys
import numpy as np
import pandas as pd
from skimage import measure
from matplotlib import pyplot as plt

import util
import kagl_preprocess
import kagl_feature_util


def extract_nodule_features(p, nodule_idx):
    r = p._nodule_regions[nodule_idx]
    local_image = p.nodule_local_image(nodule_idx)
    local_mask = p.nodule_local_mask(nodule_idx)
    local_masked_image = util.apply_mask(local_image, local_mask)
        
    props = [
        'area',
        'convex_area',
        'eccentricity',
        'equivalent_diameter',
        'major_axis_length',
        'minor_axis_length',
        'orientation',
        'perimeter',
        'solidity'
    ]
    df = []
    for z in range(len(local_mask)):
        z_region = measure.regionprops(local_mask[z].astype(int))[0]
        df.append([z_region[prop] for prop in props])
    df = pd.DataFrame(df, columns=props)

    total_area = float(np.sum(df['area']))
    ans = [
        r.area,
        len(df),
        total_area / len(df)
    ] + [
        np.sum(df[prop] * df['area']) * 1.0 / total_area
        for prop in props[1:]
    ] + (kagl_feature_util.extract_array_features(local_image) +
         kagl_feature_util.extract_array_features(local_masked_image))
    return ans


def extract_patient_features(stage, patient_name):
    p = kagl_feature_util.Patient(stage, patient_name)
    
    MIN_D, MAX_D = 3.0, 33.0
    f = []
    for nodule_idx, r in enumerate(p._nodule_regions):
        if (r.area < util.ball3d_volume(MIN_D/2.0) or
            r.area > util.ball3d_volume(MAX_D/2.0) or
            any(np.array(r.filled_image.shape)<=1)):
            continue
        f.append(extract_nodule_features(p, nodule_idx))
    f = np.asarray(f, dtype=float)

    ans = [len(f)]
    for i in range(f.shape[1]):
        ans += kagl_feature_util.extract_array_features(f[:,i])
    return ans


if __name__ == '__main__':
    stage = 'stage1'
    meta_patient = kagl_preprocess.MetaPatient(stage)
    ans = {}
    for name, label in meta_patient.labels.iteritems():
        print '========== Process %s: %d of %d'%(
            name, len(ans) + 1, len(meta_patient.labels))
        sys.stdout.flush()
        ans[name] = extract_patient_features(stage, name)
    np.save('kagl_output_feature_set1.npy', ans)
