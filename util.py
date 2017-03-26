import numpy as np

from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from skimage import measure
from skimage import morphology
from skimage import segmentation
from skimage import filters
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def find_first_neq(a, val):
    for i, x in enumerate(a):
        if np.any(np.abs(x - val) > 1.0e-6):
            return i


def find_bbox(a, margin, bg):
    margin = np.round(margin).astype(np.int)
    slices = []
    for axis in range(len(a.shape)):
        view = np.rollaxis(a, axis)
        dim = a.shape[axis]
        m = margin[axis]
        front = find_first_neq(view, bg)
        front = min(dim, max(0, front - m))
        back = find_first_neq(view[::-1], bg)
        back = min(dim, max(0, back - m))
        slices.append(slice(front, dim - back))
    return slices


def resample(image, spacing, new_spacing=(1.0, 1.0, 1.0)):
    """ Resample image from spacing to new_spacing.

    Args:
      image: 3D image of shape (z, y, x).
      spacing: (x, y, z) spacing of image.
      new_spacing: resampling to this new spacing.

    Returns:
      Image resampled to new_spacing.
    """
    spacing = np.asarray(spacing, dtype=np.float)
    new_spacing = np.asarray(new_spacing, dtype=np.float)
    
    resize_factor = (spacing / new_spacing)
    image = ndimage.interpolation.zoom(image, zoom=resize_factor[::-1], mode='nearest')

    return image


def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def _dilate_mask(mask, spacing):
    return morphology.binary_dilation(mask, morphology.ball(10.0 / spacing))


def _largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    mask = (vals!=bg)
    counts = counts[mask]
    vals = vals[mask]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask_v1(image, spacing):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label==labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if True:  # fill_lung_structures
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = _largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = _largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels!=l_max] = 0

    return _dilate_mask(binary_image, spacing)


def _postprocess_mask(mask):
    binary = mask

    labels = measure.label(binary)    
    vals, counts = np.unique(labels, return_counts=True)
    counts = counts[vals != 0]
    vals = vals[vals != 0]
    counts_and_vals = sorted(zip(counts, vals), key=lambda x: -x[0])
    
    if len(counts) == 0:
        return None
    
    max_label_counts, max_label = counts_and_vals[0]
    if len(counts) == 1:
        binary[labels!=max_label] = 0
        return binary
    
    max2_label_counts, max2_label = counts_and_vals[1]
    if max2_label_counts > 0.2 * max_label_counts:
        binary[(labels!=max_label)&(labels!=max2_label)] = 0
    else:
        binary[labels!=max_label] = 0
    return binary


def _segment_lung_mask_v2_impl(im, spacing, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -400
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = segmentation.clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = measure.label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = morphology.disk(2.0 / spacing)
    binary = morphology.binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = morphology.disk(10.0 / spacing)
    binary = morphology.binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = filters.roberts(binary)
    binary = ndimage.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 

    return binary


def segment_lung_mask_v2(image, spacing):
    binary = np.stack([
        _segment_lung_mask_v2_impl(im, spacing)
        for im in image
    ])
    return _dilate_mask(_postprocess_mask(binary), spacing)


def _segment_lung_mask_v3_impl(img, spacing):
    SCALE = 0.742188 / spacing
    img_h, img_w = img.shape
    #Standardize the pixel values
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    pad = int(round(100 * SCALE))
    middle = img[pad:img_h-pad, pad:img_w-pad]
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (radio-opaque tissue)
    # and background (radio transparent tissue ie lungs)
    # Doing this only on the center of the image to avoid 
    # the non-tissue parts of the image as much as possible
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    #
    # I found an initial erosion helful for removing graininess from some of the regions
    # and then large dialation is used to make the lung region 
    # engulf the vessels and incursions into the lung cavity by 
    # radio opaque tissue
    #
    eroded = morphology.erosion(thresh_img, morphology.square(4.0 / spacing))
    dilation = morphology.dilation(eroded, morphology.square(10.0 / spacing))
    #
    #  Label each region and obtain the region properties
    #  The background region is removed by removing regions 
    #  with a bbox that is to large in either dimnsion
    #  Also, the lungs are generally far away from the top 
    #  and bottom of the image, so any regions that are too
    #  close to the top and bottom are removed
    #  This does not produce a perfect segmentation of the lungs
    #  from the image, but it is surprisingly good considering its
    #  simplicity. 
    #
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if (B[2] - B[0] < 475 * SCALE and
            B[3] - B[1] < 475 * SCALE and
            B[0] > 40 * SCALE and B[2] < 472 * SCALE):
            good_labels.append(prop.label)
    mask = np.ndarray([img_h, img_w],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    return mask.astype(np.bool)


def segment_lung_mask_v3(image, spacing):
    binary = np.stack([
        _segment_lung_mask_v3_impl(im, spacing)
        for im in image
    ])
    return _dilate_mask(_postprocess_mask(binary), spacing)


def normalize(image, pixel_mean=0.25):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image - pixel_mean


def plot_ct_scan(image):
    f, plots = plt.subplots(int(image.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, image.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(image[i], cmap=plt.cm.bone)


def apply_argsort(a, p):
    return a[list(np.ogrid[[slice(x) for x in a.shape]][:-1])+[p]]


def _enhance_filter_2d_impl(image, sigma, kind):
    bg = image[0, 0]
    fyy = gaussian_filter(image, sigma, order=(2, 0) , mode='constant', cval=bg)
    fxx = gaussian_filter(image, sigma, order=(0, 2) , mode='constant', cval=bg)
    fxy = gaussian_filter(image, sigma, order=(1, 1) , mode='constant', cval=bg)
    a = np.zeros(list(image.shape) + [2, 2])
    a[:,:,0,0] = fxx
    a[:,:,1,1] = fyy
    a[:,:,1,0] = a[:,:,0,1] = fxy
    w, _ = np.linalg.eig(a)
    w = apply_argsort(w, np.argsort(np.abs(w)))
    lambda1 = w[:,:,1]
    lambda2 = w[:,:,0]
    if kind == 'dot':
        ans = np.abs(lambda2)**2 / np.abs(lambda1)
        ans[~((lambda1<0)&(lambda2<0))] = 0.0
    elif kind == 'line':
        ans = np.abs(lambda1) - np.abs(lambda2)
        ans[~(lambda1<0)] = 0.0
    ans[~np.isfinite(ans)] = 0.0
    return ans * (sigma**2)


def enhance_filter_2d(image, sigmas, kind='dot'):
    ans = None
    for sigma in sigmas:
        tmp = _enhance_filter_2d_impl(image, sigma, kind)
        if ans is None:
            ans = tmp
        else:
            ans = np.maximum(ans, tmp)
    return ans


def _enhance_filter_3d_impl(image, sigma, kind):
    bg = image[0, 0, 0]
    fxx = gaussian_filter(image, sigma, order=(0, 0, 2) , mode='constant', cval=bg)
    fyy = gaussian_filter(image, sigma, order=(0, 2, 0) , mode='constant', cval=bg)
    fzz = gaussian_filter(image, sigma, order=(2, 0, 0) , mode='constant', cval=bg)
    fxy = gaussian_filter(image, sigma, order=(0, 1, 1) , mode='constant', cval=bg)
    fyz = gaussian_filter(image, sigma, order=(1, 1, 0) , mode='constant', cval=bg)
    fzx = gaussian_filter(image, sigma, order=(1, 0, 1) , mode='constant', cval=bg)
    a = np.zeros(list(image.shape) + [3, 3])
    a[:,:,:,0,0] = fxx
    a[:,:,:,1,1] = fyy
    a[:,:,:,2,2] = fzz
    a[:,:,:,0,1] = a[:,:,:,1,0] = fxy
    a[:,:,:,1,2] = a[:,:,:,2,1] = fyz
    a[:,:,:,0,2] = a[:,:,:,2,0] = fzx
    w, _ = np.linalg.eig(a)
    w = apply_argsort(w, np.argsort(np.abs(w)))
    lambda1 = w[:,:,:,2]
    lambda2 = w[:,:,:,1]
    lambda3 = w[:,:,:,0]
    if kind == 'dot':
        ans = np.abs(lambda3)**2 / np.abs(lambda1)
        ans[~((lambda1<0)&(lambda2<0)&(lambda3<0))] = 0.0
    elif kind == 'line':
        ans = np.abs(lambda2) * (np.abs(lambda2) - np.abs(lambda3)) / np.abs(lambda1)
        ans[~((lambda1<0)&(lambda2<0))] = 0.0
    elif kind == 'plane':
        ans = np.abs(lambda1) - np.abs(lambda2)
        ans[~(lambda1<0)] = 0.0
    ans[~np.isfinite(ans)] = 0.0
    return ans * (sigma**2)


def enhance_filter_3d(image, sigmas, kind='dot'):
    ans = None
    for sigma in sigmas:
        tmp = _enhance_filter_3d_impl(image, sigma, kind)
        if ans is None:
            ans = tmp
        else:
            ans = np.maximum(ans, tmp)
    return ans


def get_dot_enhance_filter_sigmas(d0, d1, N=5):
    r = np.power(d1 / d0, 1.0 / (N - 1))
    sigmas = np.power(r, np.arange(N)) * d0 / 4
    return sigmas



def clip_and_pad(image, new_shape):
    assert len(image.shape)==2

    bg = image[0, 0]

    h, w = image.shape
    new_h, new_w = new_shape
    
    ph = max(0, h-new_h)
    pw = max(0, w-new_w)
    image = image[ph//2:(h-(ph-ph//2)), pw//2:(w-(pw-pw//2))]
    
    h, w = image.shape
    ph = new_h-h
    pw = new_w-w
    image = np.pad(image, ((ph//2, ph-ph//2), (pw//2, pw-pw//2)),
                   mode='constant', constant_values=bg)
    return image


def pad_to_square(image):
    assert len(image.shape)==2

    h, w = image.shape
    return clip_and_pad(image, (max(h, w), max(h, w)))


def clip_dim0(image, z):
    assert len(image.shape) == 3
    return np.clip(z, 0, image.shape[0] - 1)


def shuffle(arr, max_n=None, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    if max_n is None:
        max_n = len(arr)
    perm_idxes = random_state.permutation(len(arr))[0:max_n]
    return [arr[i] for i in perm_idxes]


def to_bool_mask(mask):
    return mask>=0.5


def is_pos_mask(mask):
    return np.sum(to_bool_mask(mask))>=0.5


def apply_mask(image, mask):
    image = image.copy()
    image[to_bool_mask(mask)==False] = -1000
    return image
