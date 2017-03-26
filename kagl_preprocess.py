import os
import sys
import csv
import numpy as np
import dicom

import util


_DATA_DIR = '../KAGL16'


def get_output_dir(stage):
    return os.path.join(_DATA_DIR, stage + '_output')


def _read_patient_names(stage_dir):
    return os.listdir(os.path.join(stage_dir))


def _read_patient_labels(csv_fname):
    with open(csv_fname) as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()  # skip header
        return dict([(r[0], float(r[1])) for r in reader if len(r) == 2])


class MetaPatient(object):

    def __init__(self, stage):
        self._stage = stage
        self._stage_dir = os.path.join(_DATA_DIR, stage)
        self._labels_csv = os.path.join(_DATA_DIR, stage + '_labels.csv')
        self._sample_csv = os.path.join(_DATA_DIR, stage + '_sample_submission.csv')
        names = _read_patient_names(self._stage_dir)
        labels = _read_patient_labels(self._labels_csv)
        test_names = set(_read_patient_labels(self._sample_csv).keys())
        self._labels = {}
        for name in names:
            if name in labels:
                self._labels[name] = labels[name]
            elif name in test_names:
                self._labels[name] = None
        self._test_names = list(test_names)
        
    @property
    def labels(self):
        return self._labels

    @property
    def test_names(self):
        return self._test_names


def _load_scans(stage, patient_name):
    path = os.path.join(_DATA_DIR, stage, patient_name)
    slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                 slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation -
                                 slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def _get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    outside_idx = (image == -2000)
    
    # Convert to Hounsfield units (HU)
    intercept = float(scans[0].RescaleIntercept)
    slope = float(scans[0].RescaleSlope)
    
    image = (slope * image + intercept).astype(np.int16)
    image[outside_idx] = -1000  # HU of air
    
    return image


def _get_spacing(scan):
    # x, y, z
    ans = [scan.PixelSpacing[0], scan.PixelSpacing[1], scan.SliceThickness]
    return [float(s) for s in ans]


class Image(object):
    
    def __init__(self, stage):
        self._stage = stage

    def init(self,
             patient_name,
             patient_label,
             iso_spacing=1.0,  # mm
             shrink_margin=2.0):  # mm
        self._patient_name = patient_name
        self._patient_label = patient_label

        scans = _load_scans(self._stage, patient_name)
        self._image = _get_pixels_hu(scans)
        self._spacing = _get_spacing(scans[0])

        self._iso_resample(iso_spacing)
        self._make_lung_mask()
        self._shrink(shrink_margin)

    def save(self, output_dir=None):
        if output_dir is None:
            output_dir = get_output_dir(self._stage)
        np.savez_compressed(
            os.path.join(output_dir, self._patient_name + '_digest'),
            patient_name=self._patient_name,
            patient_label=self._patient_label,
            spacing=self._spacing,
            image=self._image,
            lung_mask=self._lung_mask)

    def load(self, patient_name, output_dir=None):
        if output_dir is None:
            output_dir = get_output_dir(self._stage)
        ans = np.load(os.path.join(output_dir, patient_name + '_digest.npz'))
        self._patient_name = ans['patient_name'].item()
        self._patient_label = ans['patient_label'].item()
        self._spacing = ans['spacing']
        self._image = ans['image']
        self._lung_mask = ans['lung_mask']

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
        # x, y, z
        v_margin = np.round(shrink_margin / self._spacing).astype(np.int)
        slices = util.find_bbox(hero, v_margin[::-1], bg)
        self._image = self._image[slices]
        self._lung_mask = self._lung_mask[slices]

    @property
    def masked_lung(self):
        return util.apply_mask(self._image, self._lung_mask)


if __name__ == '__main__':
    stage = 'stage1'
    meta_patient = MetaPatient(stage)
    labels = meta_patient.labels

    count = 0
    for name, label in labels.iteritems():
        count += 1
        print '========== process %s: %d of %d'%(name, count, len(labels))
        sys.stdout.flush()
        try:
            image = Image(stage)
            image.init(name, label, iso_spacing=1.0, shrink_margin=2.0)
            image.save()
        except Exception as e:
            print 'Error: %s'%e
    print 'Done'
