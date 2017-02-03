import os
import sys
import csv
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob

import util
import luna_util


_DATA_DIR = '../LUNA16/subset*'
_ANNOTATION_CSV = '../LUNA16/CSVFILES/annotations.csv'
_OUTPUT_DIR = '../LUNA16/output'


def get_file_id(f):
    return os.path.basename(f)[:-4]


def get_file_list():
    return glob(os.path.join(_DATA_DIR, '*.mhd'))


def get_file_dict(file_list):
    return  {get_file_id(f) : f for f in file_list}


def get_annt_df(file_dict):
    annt_df = pd.read_csv(_ANNOTATION_CSV)
    annt_df['file'] = annt_df['seriesuid'].map(lambda suid: file_dict.get(suid))
    annt_df = annt_df.dropna()
    return annt_df


def process_file(f, annt_df, spacing, dump=False):
    itk = sitk.ReadImage(f)
    origin = itk.GetOrigin()
    old_spacing = itk.GetSpacing()

    image = sitk.GetArrayFromImage(itk)
    image_resampled = util.resample(image, old_spacing[::-1], spacing)
    num_z, height, width = image_resampled.shape

    lung_mask = util.segment_lung_mask_v2(image_resampled, spacing)

    df = annt_df[annt_df.file==f]
    d_iz = int(round(old_spacing[2] / spacing))
    nodule_masks = luna_util.extract_nodule_masks(
        num_z, height, width, spacing, [-d_iz, 0, d_iz], np.array(origin), df)

    file_id = get_file_id(f)
    for i, nodule_mask_info in enumerate(nodule_masks):
        nodule_mask, izs = nodule_mask_info
        tmp_image = image_resampled[izs]
        tmp_lung_mask = lung_mask[izs]
        if dump:
            np.save(os.path.join(_OUTPUT_DIR, "%s_image_%s.npy"%(file_id, i)),
                    tmp_image)
            np.save(os.path.join(_OUTPUT_DIR, "%s_lung_mask_%s.npy"%(file_id, i)),
                    tmp_lung_mask)
            np.save(os.path.join(_OUTPUT_DIR, "%s_nodule_mask_%s.npy"%(file_id, i)),
                    nodule_mask)
    return tmp_image, tmp_lung_mask, nodule_mask


if __name__ == '__main__':
    file_list = get_file_list()
    file_dict = get_file_dict(file_list)
    annt_df = get_annt_df(file_dict)

    for i, f in enumerate(file_list):
        print '========== process %s'%f
        sys.stdout.flush()
        process_file(f, annt_df, 1, True)
