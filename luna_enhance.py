import os
import sys
import numpy as np

import util
import luna_preprocess


_OUTPUT_DIR = '../LUNA16/output_enhance'


if __name__ == '__main__':
    file_list = luna_preprocess.get_file_list()
    file_dict = luna_preprocess.get_file_dict(file_list)
    annt_df = luna_preprocess.get_annt_df(file_dict)

    def sample_file_ids(min_num_nodules):
        t = annt_df.groupby('file').count()
        t = set([luna_preprocess.get_file_id(x)
                 for x in list(t[t.seriesuid>min_num_nodules].index)])
        # len('_digest.npz') == 11
        t1 = set([x[0:-11] for x in os.listdir(luna_preprocess._OUTPUT_DIR)])
        # len('_dot.npz') == 8
        t2 = set([x[0:-8] for x in os.listdir(_OUTPUT_DIR)])
        return list(t.intersection(t1) - t2)

    file_ids = sample_file_ids(1)

    sigmas = util.get_dot_enhance_filter_sigmas(d0=3, d1=33, N=5)

    for f_id in file_ids:
        print '========== process %s of %d'%(f_id, len(file_ids))
        sys.stdout.flush()
        image = luna_preprocess.Image()
        image.load(f_id)
        enhance_3d = util.enhance_filter_3d(image.masked_lung, sigmas, 'dot')
        np.savez_compressed(os.path.join(_OUTPUT_DIR, f_id + '_dot'),
                            enhance_3d=enhance_3d)
    print 'Done'
