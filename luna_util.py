import numpy as np
import pandas as pd


def make_nodule_mask(width, height, spacing, z, origin, center, diam):
    """Makes nodule mask for LUNA dataset.
  
    Args:
      width: width of image.
      height: height of image.
      spacing: (isotropic) pixel spacing (mm).
      z: image z (mm).
      origin: reference origin x, y, z (mm).
      center: nodule center x, y, z (mm).
      diam: nodule diameter (mm).
    """
    mask = np.zeros([height,width], dtype=np.bool)
    # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    origin = np.asarray(origin)
    center = np.asarray(center)

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin) / spacing
    pad = 5
    v_diam = int(diam / spacing + pad)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-pad])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+pad])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-pad]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+pad])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [i*spacing+origin[0] for i in range(width)]
    y_data = [i*spacing+origin[1] for i in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing*v_x + origin[0]
            p_y = spacing*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing),int((p_x-origin[0])/spacing)] = True
    return mask


def extract_nodule_masks(num_z, height, width, spacing, d_izs, origin, nodule_df):
    origin = np.asarray(origin)
    ans = []
    for node_idx, cur_row in nodule_df.iterrows():
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]
        center = np.array([node_x, node_y, node_z])
        v_center = np.rint((center-origin)/spacing)

        masks = []
        izs = (np.asarray(d_izs) + v_center[2]).astype(np.int).clip(0, num_z-1)
        for i, iz in enumerate(izs):
            masks.append(make_nodule_mask(
                width, height, spacing, iz*spacing+origin[2], origin, center, diam))
        ans.append((np.stack(masks), izs))
    return ans