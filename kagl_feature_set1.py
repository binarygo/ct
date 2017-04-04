import sys
import numpy as np
import pandas as pd
from skimage import measure
from matplotlib import pyplot as plt

import util
import kagl_preprocess
import kagl_feature_util


_MIN_NODULE_DIAMETER = 3.0
_MAX_NODULE_DIAMETER = 50.0


def extract_nodule_position_feature(p, nodule_idx):
    nod_z, nod_y, nod_x = p.nodule_position(nodule_idx)
    lung_geo = p._lung_geometry

    dz = nod_z - lung_geo._center_z
    if dz >= 0.0:
        dz_scale = abs(lung_geo._max_dz)
    else:
        dz_scale = abs(lung_geo._min_dz)
    dz = dz * 1.0 / dz_scale if dz_scale > 0 else 0.0

    r = util.distance(nod_x, nod_y, lung_geo._center_x, lung_geo._center_y)
    ay, ax = lung_geo.z_section(int(nod_z))
    ad = util.distance(ax, ay, lung_geo._center_x, lung_geo._center_y)
    r_scale = np.max(ad)
    r = r * 1.0 / r_scale if r_scale > 0 else 0.0

    br = np.min(ad)
    br = br * 1.0 / r_scale if r_scale > 0 else 0.0

    return dz, r, br


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
    ] + list(extract_nodule_position_feature(p, nodule_idx)) + [
        np.mean(local_masked_image),
        np.std(local_masked_image)
    ] + [
        kagl_feature_util.nodule_boundary_diff(p, nodule_idx)
    ] + [
        np.sum(df[prop] * df['area']) * 1.0 / total_area
        for prop in props[1:]
    ]
    return ans


def extract_patient_features(stage, patient_name):
    p = kagl_feature_util.Patient(stage, patient_name)
    
    nodule_features = []
    for nodule_idx, r in enumerate(p._nodule_regions):
        if (r.area < util.ball3d_volume(_MIN_NODULE_DIAMETER/2.0) or
            r.area > util.ball3d_volume(_MAX_NODULE_DIAMETER/2.0) or
            any(np.array(r.filled_image.shape)<=1)):
            continue
        nodule_features.append(extract_nodule_features(p, nodule_idx))
    nodule_features = np.asarray(nodule_features, dtype=float)

    lung_features = [
        np.sum(p._lung_mask),
        np.mean(p._masked_lung),
        np.std(p._masked_lung),
        p._lung_geometry._min_dz,
        p._lung_geometry._max_dz,
        p._lung_geometry._min_r,
        p._lung_geometry._max_r,
        p._lung_geometry._mean_r,
        p._lung_geometry._std_r,
        p.num_nodules
    ]

    all_features = lung_features[:]
    for i in range(nodule_features.shape[1]):
        all_features += kagl_feature_util.extract_array_features(nodule_features[:,i])
    return all_features, (lung_features, nodule_features)


if __name__ == '__main__':
    stage = 'stage1'
    meta_patient = kagl_preprocess.MetaPatient(stage)
    ans_all_features = {}
    ans_raw_features = {}
    for name, label in meta_patient.labels.iteritems():
        print '========== Process %s: %d of %d'%(
            name, len(ans_all_features) + 1, len(meta_patient.labels))
        sys.stdout.flush()
        all_features, raw_features = extract_patient_features(stage, name)
        ans_all_features[name] = all_features
        ans_raw_features[name] = raw_features
        print 'num_features = %d'%len(ans_all_features[name])
    np.save('kagl_output_feature_set1.npy', ans_all_features)
    np.save('kagl_output_raw_feature_set1.npy', ans_raw_features)
