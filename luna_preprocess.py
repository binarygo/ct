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

def process_file(f, annt_df):
    itk = sitk.ReadImage(f)
    image = sitk.GetArrayFromImage(itk)
    spacing = 1
    image_resampled = util.resample(image, itk.GetSpacing()[::-1], spacing)

    lung_mask = util.segment_lung_mask_v2(image_resampled, spacing)
    lung_image = util.apply_mask(image_resampled, lung_mask)

    df = annt_df[annt_df.file==f]
    nodule_masks = luna_util.extract_nodule_masks(
        lung_image, spacing, np.array(itk.GetOrigin()), df)

    file_id = get_file_id(f)
    for i, nodule_mask in enumerate(nodule_masks):
        tmp_image, tmp_nodule_mask = nodule_mask
        np.save(os.path.join(_OUTPUT_DIR, "%s_image_%s.npy"%(file_id, i)), tmp_image)
        np.save(os.path.join(_OUTPUT_DIR, "%s_nodule_mask_%s.npy"%(file_id, i)), tmp_nodule_mask)
        

if __name__ == '__main__':
    file_list = glob(os.path.join(_DATA_DIR, '*.mhd'))
    file_dict = {get_file_id(f) : f for f in file_list}

    annt_df = pd.read_csv(_ANNOTATION_CSV)
    annt_df['file'] = annt_df['seriesuid'].map(lambda suid: file_dict.get(suid))
    annt_df = annt_df.dropna()

    for i, f in enumerate(file_list):
        print '========== process %s'%f
        sys.stdout.flush()
        process_file(f, annt_df)
