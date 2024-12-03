import math
import networkx as nx
from scipy.spatial.distance import cdist
import numpy as np




def get_shortest_path_from_tip_to_sources(graph,sources,tip):
    paths = []
    for source in sources:
        if nx.has_path(graph,tip,source):
            path = nx.shortest_path(graph,tip,source)
            paths.append(path)
    #get the shortest path
    shortest_path = min(paths,key=len)
    return shortest_path

#walk to a neighbor node
def walk_to_neighbor(graph,tip_root_path):
    current_node = tip_root_path.get_current_path_last_node()
    walked_nodes = tip_root_path.get_current_path_walked_nodes()
    
    neighbors = list(graph.neighbors(current_node))
    #remove the nodes that have been visited
    neighbors = [node for node in neighbors if node not in walked_nodes]
    #if the current point y is over the neighbor node + threshold, remove the neighbor node
    threshold = 10
    #get current node coords
    current_x, current_y = graph.nodes[current_node]['coord'][1], graph.nodes[current_node]['coord'][0]
    neighbors = [node for node in neighbors if graph.nodes[node]['coord'][0] < current_y + threshold]

    #check if some sources are in the neighbors
    sources = [node for node in neighbors if graph.nodes[node]['label'] == 'source']
        
    return neighbors,sources

def calculate_angle(p1, p2):
    """
    Calculate the angle between two points in an image where to the left of the point the angle is < 0,-90 and to the right is >0,90.

    Args:
        p1 (tuple): The first point (x, y).
        p2 (tuple): The second point (x, y).

    Returns:
        float: The angle between the two points in degrees.
    """
    delta_x = p2[1] - p1[1]
    delta_y = p2[0] - p1[0]
    
    angle = math.degrees(math.atan2(delta_y, delta_x))
    
    # # Adjust the angle based on the relative positions of the points
    # if delta_x < 0:
    #     angle -= 180
    # elif delta_x == 0 and delta_y < 0:
    #     angle = -90
    # elif delta_x == 0 and delta_y > 0:
    #     angle = 90

    return angle

from scipy.spatial import cKDTree
import numpy as np

def find_nearest_tip(predicted_tip, gt_tips, max_range=50):
    """
    Find the nearest ground truth tip to the predicted tip within a specified range.
    """
    if not gt_tips:
        return None, float('inf')
    
    gt_tips_flipped = [(y, x) for x, y in gt_tips]  # Invert the coordinates
    tree = cKDTree(gt_tips_flipped)
    distance, index = tree.query(predicted_tip, distance_upper_bound=max_range)
    
    if distance == float('inf'):
        return None, distance  # No tip found within the range
    return index+1, distance

# def compare_paths(predicted_path, gt_path):
#     """
#     Compare two paths by calculating the normalized average distance between ground truth points and their closest points in the predicted path.

#     Args:
#         predicted_path (list): List of (x, y) coordinates in the predicted path.
#         gt_path (list): List of (x, y) coordinates in the ground truth path.

#     Returns:
#         float: Normalized average distance.
#         list: List of distances between GT points and closest predicted points.
#     """
#     if not predicted_path or not gt_path:
#         return float('inf'), []  # Invalid comparison if paths are missing

#     # Convert paths to NumPy arrays for distance calculations
#     pred_array = np.array(predicted_path)
#     gt_array = np.array(gt_path)

#     # Calculate pairwise distances between GT points and predicted path points
#     distances = cdist(gt_array, pred_array, metric='euclidean')

#     # Find the closest predicted point for each GT point
#     min_distances = distances.min(axis=1)

#     # Normalize the distances (e.g., by max GT distance or total path length)
#     max_gt_distance = np.linalg.norm(gt_array[-1] - gt_array[0])  # Length of GT path
#     if max_gt_distance == 0:  # Avoid division by zero for very short paths
#         max_gt_distance = 1.0
#     normalized_distance = np.mean(min_distances) / max_gt_distance

#     return normalized_distance, min_distances.tolist()
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

def resample_path(path, num_points):
    """
    Resample a path to have a specified number of equidistant points.

    Args:
        path (np.ndarray): Array of (x, y) coordinates.
        num_points (int): Number of points in the resampled path.

    Returns:
        np.ndarray: Resampled path with shape (num_points, 2).
    """
    path = np.array(path)
    # Compute cumulative distances along the path
    distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    # Create interpolation functions for x and y coordinates
    f_interp_x = interp1d(cumulative_distances, path[:, 0], kind='linear')
    f_interp_y = interp1d(cumulative_distances, path[:, 1], kind='linear')
    # Generate new distances
    new_distances = np.linspace(0, cumulative_distances[-1], num=num_points)
    # Interpolate to get new points
    resampled_x = f_interp_x(new_distances)
    resampled_y = f_interp_y(new_distances)
    resampled_path = np.stack((resampled_x, resampled_y), axis=-1)
    return resampled_path

def compute_angles(path):
    """
    Compute angles between consecutive points in a path.

    Args:
        path (np.ndarray): Array of (x, y) coordinates.

    Returns:
        np.ndarray: Array of angles in radians.
    """
    deltas = np.diff(path, axis=0)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])  # Assuming (x, y) coordinates
    return angles

def extract_nodes(path, node_threshold=5):
    """
    Extract key nodes (e.g., corners) from a path based on curvature.

    Args:
        path (np.ndarray): Array of (x, y) coordinates.
        node_threshold (float): Threshold for detecting nodes.

    Returns:
        np.ndarray: Array of node coordinates.
    """
    # Compute angles between segments
    angles = compute_angles(path)
    angle_changes = np.abs(np.diff(angles))
    # Identify points where the angle change exceeds the threshold
    node_indices = np.where(angle_changes > np.deg2rad(node_threshold))[0] + 1
    # Include the first and last points as nodes
    node_indices = np.insert(node_indices, 0, 0)
    node_indices = np.append(node_indices, len(path) - 1)
    nodes = path[node_indices]
    return nodes

def compute_node_matching_score(gt_nodes, pred_nodes, threshold=10):
    """
    Compute a node matching score between GT nodes and predicted nodes.

    Args:
        gt_nodes (np.ndarray): Array of GT node coordinates.
        pred_nodes (np.ndarray): Array of predicted node coordinates.
        threshold (float): Distance threshold for matching nodes.

    Returns:
        float: Node matching score between 0 and 1.
    """
    if len(gt_nodes) == 0 or len(pred_nodes) == 0:
        return 0.0
    # Compute pairwise distances between nodes
    distances = cdist(gt_nodes, pred_nodes)
    # Determine matches within the threshold
    matches = distances <= threshold
    # Count matched GT nodes
    matched_gt_nodes = np.any(matches, axis=1)
    num_matched_gt_nodes = np.sum(matched_gt_nodes)
    # Calculate the node matching score
    node_matching_score = num_matched_gt_nodes / len(gt_nodes)
    return node_matching_score

def compare_paths(predicted_path, gt_path):
    """
    Compare two paths using multiple metrics:
    - Normalized average point-wise distance : Misura la distanza media tra i punti corrispondenti dei due percorsi, normalizzata per la lunghezza totale del percorso GT.
                                               Un valore vicino a 0 indica che il percorso predetto segue molto da vicino il percorso GT.
                                               Valori inferiori a 0,1 (cioè, la distanza media è meno del 10% della lunghezza totale del percorso GT).
    
    - Normalized maximum point-wise distance: Misura la distanza massima tra i punti corrispondenti dei due percorsi, normalizzata per la lunghezza totale del percorso GT
                                                Indica la peggiore discrepanza tra i due percorsi. Valori bassi suggeriscono che non ci sono deviazioni significative in nessun punto
    - Average angular difference :Misura la differenza media degli angoli tra i segmenti corrispondenti dei due percorsi, espressa in radianti
                                    Valori bassi indicano che i due percorsi hanno una direzione simile lungo tutta la loro lunghezza
    - Node matching score : Percentuale di nodi del percorso GT che trovano una corrispondenza nel percorso predetto entro una certa soglia di distanza
                            Un punteggio elevato indica che la struttura fondamentale del percorso (ad esempio, curve e biforcazioni) è stata catturata accuratamente
    Args:
        predicted_path (list): List of (x, y) coordinates in the predicted path.
        gt_path (list): List of (x, y) coordinates in the ground truth path.

    Returns:
        dict: Dictionary containing comparison metrics.
    """
    if not predicted_path or not gt_path:
        return {
            'normalized_average_distance': float('inf'),
            'normalized_maximum_distance': float('inf'),
            'average_angular_difference': float('inf'),
            'node_matching_score': 0.0,
            
        }

    # Convert paths to NumPy arrays
    pred_array = np.array(predicted_path)
    gt_array = np.array(gt_path)

    # Compute total length of the GT path
    gt_path_distances = np.linalg.norm(np.diff(gt_array, axis=0), axis=1)
    total_gt_path_length = np.sum(gt_path_distances)
    if total_gt_path_length == 0:
        total_gt_path_length = 1.0  # Avoid division by zero

    # Resample paths to have the same number of points
    num_points = max(len(gt_array), len(pred_array))
    gt_resampled = resample_path(gt_array, num_points)
    pred_resampled = resample_path(pred_array, num_points)

    # Compute point-wise distances
    pointwise_distances = np.linalg.norm(gt_resampled - pred_resampled, axis=1)
    average_distance = np.mean(pointwise_distances)
    maximum_distance = np.max(pointwise_distances)

    # Normalize distances
    normalized_average_distance = average_distance / total_gt_path_length
    normalized_maximum_distance = maximum_distance / total_gt_path_length

    # Compute angles for both paths
    gt_angles = compute_angles(gt_resampled)
    pred_angles = compute_angles(pred_resampled)

    # Ensure angles are within [-π, π]
    gt_angles = np.unwrap(gt_angles)
    pred_angles = np.unwrap(pred_angles)

    # Resample angles to have the same number of points
    num_angles = min(len(gt_angles), len(pred_angles))
    gt_angles_resampled = np.interp(
        np.linspace(0, len(gt_angles) - 1, num_angles),
        np.arange(len(gt_angles)),
        gt_angles
    )
    pred_angles_resampled = np.interp(
        np.linspace(0, len(pred_angles) - 1, num_angles),
        np.arange(len(pred_angles)),
        pred_angles
    )

    # Compute angular differences
    angular_differences = np.abs(gt_angles_resampled - pred_angles_resampled)
    # Adjust angles to be within [0, π]
    angular_differences = np.mod(angular_differences, np.pi)
    average_angular_difference = np.mean(angular_differences)

    # Extract nodes from paths
    gt_nodes = extract_nodes(gt_array)
    pred_nodes = extract_nodes(pred_array)

    # Compute node matching score
    node_matching_score = compute_node_matching_score(gt_nodes, pred_nodes, threshold=30)

    # Return all metrics
    return {
        'normalized_average_distance': normalized_average_distance,
        'normalized_maximum_distance': normalized_maximum_distance,
        'average_angular_difference': average_angular_difference,
        'node_matching_score': node_matching_score,
        # Add other metrics as needed
    }