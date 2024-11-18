# skeleton_utils.py

import numpy as np
from skimage.morphology import skeletonize

def get_skeleton(image):
    """
    Obtain the skeleton of a binary image.
    """
    skeleton = skeletonize(image > 0).astype(np.uint8)
    return skeleton
