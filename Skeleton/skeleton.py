
import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import convolve
from skimage.morphology import medial_axis, thin
from scipy.spatial import distance
from Skeleton import sknw
import networkx as nx
import matplotlib.pyplot as plt

def get_skeleton(mask):
    # Generate skeleton and distance transform
    skeleton, dist_trans = medial_axis(mask.astype(bool), return_distance=True)
    
    # Convert skeleton to uint8 for compatibility with OpenCV functions
    skeleton = skeleton.astype(np.uint8)
    
    # Define the cross kernel for hit-or-miss operation
    cross_ker = np.array([[-1, 1, -1],
                          [1, -1, 1],
                          [-1, 1, -1]], dtype=np.int8)
    
    # Perform morphological hit-or-miss transform using the cross kernel
    cross_nodes = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, cross_ker)
    
    # Combine skeleton and cross_nodes using bitwise OR
    skeleton = skeleton | cross_nodes
    
    # Apply skeleton thinning to further refine the skeleton
    skeleton = thin(skeleton.astype(bool)).astype(np.uint8)
    
    # Convert the final skeleton back to boolean
    skeleton = skeleton.astype(bool)
    
    return skeleton

