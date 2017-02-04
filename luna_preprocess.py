import os
import sys
import csv
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import morphology
from glob import glob

import util


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


class Image(object):
    
    def __init__(self):
        pass

    def init(self, f, annt_df,
             iso_spacing=1.0,  # mm
             shrink_margin=5.0):  # mm
        self._f = f
        self._f_id = get_file_id(f)

        itk = sitk.ReadImage(f)
        self._origin = np.asarray(itk.GetOrigin(), dtype=np.float)  # x, y, z
        self._spacing = np.asarray(itk.GetSpacing(), dtype=np.float)  # x, y, z        
        self._image = sitk.GetArrayFromImage(itk)

        self._iso_resample(iso_spacing)
        self._make_lung_mask()
        self._shrink(shrink_margin)

        nodules = []
        df = annt_df[annt_df.file==self._f]
        for _, cur_row in df.iterrows():
            nod_x = cur_row["coordX"]
            nod_y = cur_row["coordY"]
            nod_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            nodules.append((nod_x, nod_y, nod_z, diam))
        self._nodules = np.asarray(nodules, dtype=np.float)
        self._make_nodule_masks(self._nodules)

    def save(self, output_dir):
        np.savez_compressed(
            os.path.join(output_dir, self._f_id + '_digest'),
            f=self._f,
            f_id=self._f_id,
            origin=self._origin,
            spacing=self._spacing,
            image=self._image,
            lung_mask=self._lung_mask,
            nodules=self._nodules)

    def load(self, output_dir, f_id):
        ans = np.load(os.path.join(output_dir, f_id + '_digest.npz'))
        self._f = ans['f'].item()
        self._f_id = ans['f_id'].item()
        self._origin = ans['origin']
        self._spacing = ans['spacing']
        self._image = ans['image']
        self._lung_mask = ans['lung_mask']
        self._nodules = ans['nodules']
        self._make_nodule_masks(self._nodules)

    def _iso_resample(self, iso_spacing):
        new_spacing = np.asarray([iso_spacing] * 3, dtype=np.float)
        self._image = util.resample(self._image, self._spacing, new_spacing)
        self._spacing = new_spacing

    def _assert_iso_spacing(self):
        assert np.std(self._spacing) < 1.0e-6, "Needs iso spacing."

    def _make_lung_mask(self):
        self._assert_iso_spacing()
        self._lung_mask = util.segment_lung_mask_v2(self._image, self._spacing[0])

    def _shrink(self, shrink_margin):
        hero = self._lung_mask.astype(np.uint8)
        bg = hero[0, 0, 0]
        v_margin = np.round(shrink_margin / self._spacing).astype(np.int)
        offsets = []
        slices = []
        for axis in [0, 1, 2]:  # z, y, x
            view = np.rollaxis(hero, axis)
            dim = hero.shape[axis]
            m = v_margin[2 - axis]
            front = util.find_first_neq(view, bg)
            front = min(dim, max(0, front - m))
            back = util.find_first_neq(view[::-1], bg)
            back = min(dim, max(0, back - m))
            slices.append(slice(front, dim - back))
            offsets.append(front)
        self._origin = self._origin + self._spacing * offsets[::-1]
        self._image = self._image[slices]
        self._lung_mask = self._lung_mask[slices]

    def _make_nodule_mask(self, nodule):
        self._assert_iso_spacing()
        spacing = self._spacing[0]
        nod_x, nod_y, nod_z, diam = nodule
        center = np.asarray([nod_x, nod_y, nod_z], dtype=np.float)
        radius = diam / 2.0
        v_center = np.round((center - self._origin) / spacing).astype(np.int)
        v_radius = int(round(radius / spacing))
        ball = morphology.ball(v_radius)
        ans = np.zeros(self._image.shape)
        v_o = np.maximum(0, v_center - v_radius)  # x, y, z
        ans[v_o[2]:v_o[2]+ball.shape[0],
            v_o[1]:v_o[1]+ball.shape[1],
            v_o[0]:v_o[0]+ball.shape[2]] = ball
        return ans

    def _make_nodule_masks(self, nodules):
        nodule_masks = []
        for nodule in nodules:
            nodule_masks.append(self._make_nodule_mask(nodule))
        self._nodule_masks = nodule_masks

    def get_v_nodules(self):
        self._assert_iso_spacing()
        spacing = self._spacing[0]
        ans = []
        for nodule in self._nodules:
            nod_x, nod_y, nod_z, diam = nodule
            center = np.asarray([nod_x, nod_y, nod_z], dtype=np.float)
            radius = diam / 2.0
            v_center = np.round((center - self._origin) / spacing).astype(np.int)
            v_radius = int(round(radius / spacing))
            # v_x, v_y, v_z, v_diam
            ans.append(tuple(list(v_center) + [v_radius * 2]))
        return ans


if __name__ == '__main__':
    file_list = get_file_list()
    file_dict = get_file_dict(file_list)
    annt_df = get_annt_df(file_dict)

    for i, f in enumerate(file_list):
        print '========== process %s'%f
        sys.stdout.flush()
        image = Image()
        image.init(f, annt_df, iso_spacing=1.0, shrink_margin=2.0)
        image.save(_OUTPUT_DIR)
