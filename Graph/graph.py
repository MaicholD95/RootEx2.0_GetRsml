import cv2
from skimage.morphology import skeletonize
from skimage import io
import networkx as nx
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def merge_close_nodes(graph, threshold_distance):
    """
    Merge all nodes within a threshold distance from each 'intersection' node.
    
    Args:
        graph (nx.MultiGraph): The graph to modify.
        threshold_distance (float): The distance threshold within which nodes are merged.
    
    Returns:
        nx.MultiGraph: The modified graph with close nodes merged.
    """
    import numpy as np
    fused_graph = graph.copy()
    nodes_data = dict(fused_graph.nodes(data=True))
    
    # Create a list of nodes to process
    nodes_to_process = [node_id for node_id, data in fused_graph.nodes(data=True) if data.get('label') == 'intersection']
    processed_nodes = set()
    
    while nodes_to_process:
        current_node = nodes_to_process.pop(0)
        if current_node in processed_nodes:
            continue  # Skip nodes that have already been processed
        
        current_coord = nodes_data[current_node]['coord']
        nodes_to_merge = [current_node]
        
        # Find nodes within the threshold distance
        for other_node, data in fused_graph.nodes(data=True):
            if other_node == current_node or other_node in processed_nodes:
                continue
            if data.get('label') == 'intersection':
                other_coord = data['coord']
                distance = np.linalg.norm(np.array(current_coord) - np.array(other_coord))
                if distance <= threshold_distance:
                    nodes_to_merge.append(other_node)
        
        if len(nodes_to_merge) > 1:
            # Merge nodes
            fused_node_id = min(nodes_to_merge)  # Choose the smallest ID to keep
            fused_coord = fused_graph.nodes[fused_node_id]['coord']
            for node_id in nodes_to_merge:
                if node_id == fused_node_id:
                    continue
                # Redirect edges
                edges = list(fused_graph.edges(node_id, keys=True, data=True))
                for u, v, key, data in edges:
                    other_node = v if u == node_id else u
                    # Remove the old edge
                    fused_graph.remove_edge(u, v, key=key)
                    # Adjust path
                    path = data.get('path', [])
                    node_coord = fused_graph.nodes[node_id]['coord']
                    path = [fused_coord if coord == node_coord else coord for coord in path]
                    # Add new edge between fused_node_id and other_node
                    fused_graph.add_edge(fused_node_id, other_node, **data)
                    # Update path in edge data
                    edge_keys = fused_graph[fused_node_id][other_node]
                    last_key = max(edge_keys)  # Get the latest edge key
                    fused_graph[fused_node_id][other_node][last_key]['path'] = path
                # Remove the node
                fused_graph.remove_node(node_id)
                processed_nodes.add(node_id)
            # Update the coordinate of the fused node to the average position
            coords = [nodes_data[n_id]['coord'] for n_id in nodes_to_merge]
            y_coords, x_coords = zip(*coords)
            avg_coord = (sum(y_coords) / len(y_coords), sum(x_coords) / len(x_coords))
            fused_graph.nodes[fused_node_id]['coord'] = avg_coord
            nodes_data[fused_node_id]['coord'] = avg_coord
            processed_nodes.add(fused_node_id)
        else:
            processed_nodes.add(current_node)
    
    print("Completed merging of close intersection nodes.")
    return fused_graph

def divide_paths_with_equidistant_nodes(graph, interval_distance):
    """
    For each edge in the graph, divide the path into segments of equal length specified by interval_distance.
    Add new nodes at these points, label them as 'intermed', and connect them appropriately.

    Args:
        graph (nx.MultiGraph): The graph to modify.
        interval_distance (float): The distance between equidistant nodes along each path.

    Returns:
        nx.MultiGraph: The updated graph with 'intermed' nodes added.
    """
    new_graph = graph.copy()
    edges_to_remove = []

    # Initialize node ID counter
    if new_graph.nodes:
        node_id_counter = max(new_graph.nodes) + 1
    else:
        node_id_counter = 0

    # Build mapping from coordinates to node IDs
    coord_to_id = {data['coord']: node_id for node_id, data in new_graph.nodes(data=True)}

    for u, v, key, data in graph.edges(keys=True, data=True):
        path = data.get('path', [])
        if not path or len(path) < 2:
            continue  # Skip if no valid path

        # Compute cumulative distances along the path
        cumulative_distances = [0.0]
        for i in range(1, len(path)):
            y0, x0 = path[i-1]
            y1, x1 = path[i]
            dy = y1 - y0
            dx = x1 - x0
            dist = np.sqrt(dy**2 + dx**2)
            cumulative_distances.append(cumulative_distances[-1] + dist)

        total_length = cumulative_distances[-1]

        if total_length < interval_distance:
            continue  # Skip if the path is shorter than interval_distance

        # Determine the distances at which to insert nodes
        num_segments = int(total_length // interval_distance)
        segment_distances = [interval_distance * i for i in range(1, num_segments + 1)]

        new_node_ids = []
        for sd in segment_distances:
            # Find the index where sd falls in cumulative_distances
            idx = np.searchsorted(cumulative_distances, sd)
            if idx >= len(cumulative_distances):
                idx = len(cumulative_distances) - 1
            elif idx == 0:
                idx = 1  # Avoid idx-1 being -1
            # Linear interpolation to find exact position
            d0 = cumulative_distances[idx - 1]
            d1 = cumulative_distances[idx]
            ratio = (sd - d0) / (d1 - d0) if d1 != d0 else 0
            y0, x0 = path[idx - 1]
            y1, x1 = path[idx]
            y_new = y0 + ratio * (y1 - y0)
            x_new = x0 + ratio * (x1 - x0)
            # Create a new node at (y_new, x_new)
            node_coord = (int(round(y_new)), int(round(x_new)))
            if node_coord not in coord_to_id:
                node_id = node_id_counter
                node_id_counter += 1
                new_graph.add_node(node_id, label='intermed', coord=node_coord)
                coord_to_id[node_coord] = node_id
            else:
                node_id = coord_to_id[node_coord]
            new_node_ids.append(node_id)

        # Remove the original edge
        edges_to_remove.append((u, v, key))

        # Build the sequence of nodes: u, new_nodes..., v
        nodes_sequence = [u] + new_node_ids + [v]

        # Add edges between the nodes in sequence
        for i in range(len(nodes_sequence) - 1):
            node_a = nodes_sequence[i]
            node_b = nodes_sequence[i+1]
            # Find the indices in the path corresponding to node_a and node_b
            if i == 0:
                idx_a = 0
            else:
                idx_a = np.searchsorted(cumulative_distances, segment_distances[i-1])
                if idx_a >= len(path):
                    idx_a = len(path) - 1
            if i + 1 == len(new_node_ids) + 1:
                idx_b = len(path) - 1
            else:
                idx_b = np.searchsorted(cumulative_distances, segment_distances[i])
                if idx_b >= len(path):
                    idx_b = len(path) - 1
            # Get the path segment between idx_a and idx_b
            path_segment = path[idx_a:idx_b+1] if idx_a <= idx_b else path[idx_b:idx_a+1][::-1]
            # Compute the length of the segment
            length = cumulative_distances[idx_b] - cumulative_distances[idx_a]
            # Add edge between node_a and node_b
            new_graph.add_edge(node_a, node_b, weight=length, path=path_segment)

    # Remove the original edges
    for u, v, key in edges_to_remove:
        new_graph.remove_edge(u, v, key=key)

    return new_graph

def build_graph_from_skeleton(skeleton):
    """
    Constructs a NetworkX graph from a skeletonized image.
    Only points with 1 or more than 2 neighbors are considered nodes and are labeled as 'external' or 'intersection'.
    
    Args:
        skeleton (np.ndarray): Skeletonized image (binary, values 0 and 1).
    
    Returns:
        nx.MultiGraph: Graph representing the skeleton with labeled nodes.
    """
    G = nx.MultiGraph()  # Use MultiGraph to allow multiple edges between nodes
    rows, cols= skeleton.shape

    # Define offsets for 8 neighbors (8-connectivity)
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0), (1, 1)]
    
    # Compute neighbor count for each pixel using convolution
    structure = np.ones((3,3), dtype=int)
    neighbor_count = ndimage.convolve(skeleton.astype(int), structure, mode='constant', cval=0) - skeleton

    # Identify nodes
    external_nodes = set(zip(*np.where((skeleton == 1) & (neighbor_count == 1))))
    intersection_nodes = set(zip(*np.where((skeleton == 1) & (neighbor_count > 2))))
    all_nodes = external_nodes.union(intersection_nodes)

    # Initialize node ID counter and coordinate mappings
    node_id_counter = 0
    coord_to_id = {}
    id_to_coord = {}

    # Add nodes to graph with labels
    for node in external_nodes:
        node_id = node_id_counter
        node_id_counter += 1
        G.add_node(node_id, label='external', coord=node)
        coord_to_id[node] = node_id
        id_to_coord[node_id] = node
    for node in intersection_nodes:
        node_id = node_id_counter
        node_id_counter += 1
        G.add_node(node_id, label='intersection', coord=node)
        coord_to_id[node] = node_id
        id_to_coord[node_id] = node

    # Initialize a matrix to track visited pixels
    visited_pixels = np.zeros_like(skeleton, dtype=bool)

    def traverse(node_coord, direction):
        """
        Trace an edge from the skeleton in a specific direction until another node is found.
        
        Args:
            node_coord (tuple): Coordinates of the starting node (y, x).
            direction (tuple): Offset of the direction (dy, dx).
        
        Returns:
            list or None: List of coordinates composing the edge, or None if no other node is found.
        """
        path = [node_coord]
        y, x = node_coord
        dy, dx = direction
        current = (y + dy, x + dx)
        prev = node_coord

        while True:
            cy, cx = current
            if not (0 <= cy < rows and 0 <= cx < cols):
                return None  # Out of bounds
            if skeleton[cy, cx] == 0:
                return None  # Not a skeleton pixel
            if visited_pixels[cy, cx]:
                return None  # Pixel already visited
            
            path.append(current)
            visited_pixels[cy, cx] = True

            # Find neighbors of current pixel excluding previous pixel
            neighbors = []
            for offset in neighbor_offsets:
                ny, nx_ = cy + offset[0], cx + offset[1]
                neighbor = (ny, nx_)
                if (0 <= ny < rows and 0 <= nx_ < cols and
                    skeleton[ny, nx_] and neighbor != prev):
                    neighbors.append(neighbor)
            
            if len(neighbors) == 0:
                return None  # End of edge
            elif len(neighbors) > 1:
                # If current pixel is a node, end the edge
                if current in all_nodes:
                    return path
                else:
                    return None  # Ambiguity in path
            else:
                next_node = neighbors[0]
                if next_node in all_nodes:
                    path.append(next_node)
                    return path
                else:
                    prev, current = current, next_node

    # Trace edges from each node
    for node_coord in all_nodes:
        y, x = node_coord
        for offset in neighbor_offsets:
            dy, dx = offset
            neighbor = (y + dy, x + dx)
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                skeleton[neighbor] == 1 and neighbor not in all_nodes):
                if not visited_pixels[neighbor]:
                    path = traverse(node_coord, offset)
                    if path and len(path) >= 2:
                        start_node_coord = path[0]
                        end_node_coord = path[-1]
                        # Calculate the length of the edge
                        length = 0
                        for i in range(len(path)-1):
                            dy_ = path[i+1][0] - path[i][0]
                            dx_ = path[i+1][1] - path[i][1]
                            length += np.sqrt(dy_**2 + dx_**2)
                        # Add edge to graph
                        start_node_id = coord_to_id[start_node_coord]
                        end_node_id = coord_to_id[end_node_coord]
                        G.add_edge(start_node_id, end_node_id, weight=length, path=path)

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def prune_external_nodes(graph, min_length=30):
    """
    Prune 'external' nodes and their edges if the edge lengths are less than min_length.
    This function is applied recursively until no 'external' nodes with short edges remain.

    Args:
        graph (nx.MultiGraph): The graph to clean.
        min_length (float): Minimum length of edges to retain.

    Returns:
        nx.MultiGraph: The cleaned graph without the 'external' nodes and their edges.
    """
    pruned_graph = graph.copy()
    nodes_removed = True
    iteration = 0

    while nodes_removed:
        nodes_removed = False
        iteration += 1

        # Identify 'external' nodes
        external_nodes = [
            node for node, data in pruned_graph.nodes(data=True)
            if data.get('label') == 'external'
        ]

        # Identify 'external' nodes with edges shorter than min_length
        nodes_to_remove = []
        for node in external_nodes:
            # 'External' nodes should have only one edge
            edges = list(pruned_graph.edges(node, keys=True, data=True))
            if len(edges) == 0:
                nodes_to_remove.append(node)  # Isolated node, remove
                continue
            for edge in edges:
                length = edge[3].get('weight', 0)
                if length < min_length:
                    nodes_to_remove.append(node)
                    break  # Remove the node if any of its edges is < min_length

        if nodes_to_remove:
            print(f"Removing {len(nodes_to_remove)} external nodes with edges shorter than {min_length}.")
            # Remove the identified nodes
            pruned_graph.remove_nodes_from(nodes_to_remove)
            nodes_removed = True
        else:
            print("No more external nodes to remove.")

    return pruned_graph

def merge_degree_two_intersections(graph):
    """
    Transform 'intersection' nodes with exactly 2 neighbors into direct edges between their neighbors,
    following the skeleton path.

    Args:
        graph (nx.MultiGraph): The graph to modify.

    Returns:
        nx.MultiGraph: The modified graph without the 'intersection' nodes with degree 2.
    """
    merged = True
    while merged:
        merged = False
        # Identify 'intersection' nodes with degree exactly 2
        nodes_to_merge = [
            node for node, data in graph.nodes(data=True)
            if data.get('label') == 'intersection' and graph.degree(node) == 2
        ]
        
        if not nodes_to_merge:
            break  # No nodes to merge
        
        for node in nodes_to_merge:
            neighbors = list(graph.neighbors(node))
            if len(neighbors) != 2:
                continue  # Skip if not exactly 2 neighbors
            
            u, v = neighbors
            # Get all edges between node and u, and node and v
            edges_u = list(graph.get_edge_data(node, u, default={}).items())  # List of (key, data_dict)
            edges_v = list(graph.get_edge_data(node, v, default={}).items())
            
            # Ensure there's at least one edge between node and u and node and v
            if not edges_u or not edges_v:
                continue
            
            # Take the first available edge between node and u and node and v
            key_u, data_u = edges_u[0]
            key_v, data_v = edges_v[0]
            
            path_u = data_u.get('path', [])
            path_v = data_v.get('path', [])
            
            # Avoid merging edges without valid paths
            if not path_u or not path_v:
                continue
            
            # Verify path direction
            node_coord = graph.nodes[node]['coord']
            u_coord = graph.nodes[u]['coord']
            v_coord = graph.nodes[v]['coord']

            if path_u[0] == u_coord and path_u[-1] == node_coord:
                ordered_path_u = path_u
            elif path_u[0] == node_coord and path_u[-1] == u_coord:
                ordered_path_u = path_u[::-1]
            else:
                continue  # Invalid path
            
            if path_v[0] == node_coord and path_v[-1] == v_coord:
                ordered_path_v = path_v
            elif path_v[0] == v_coord and path_v[-1] == node_coord:
                ordered_path_v = path_v[::-1]
            else:
                continue  # Invalid path
            
            # Build the new path including the intermediate node
            new_path = ordered_path_u[:-1] + [node_coord] + ordered_path_v[1:]
            
            # Calculate the new length
            length = 0
            for i in range(len(new_path)-1):
                dy = new_path[i+1][0] - new_path[i][0]
                dx = new_path[i+1][1] - new_path[i][1]
                length += np.sqrt(dy**2 + dx**2)
            
            # Add the new edge between u and v
            graph.add_edge(u, v, weight=length, path=new_path)
            
            # Remove the original edges using the correct keys
            graph.remove_edge(node, u, key=key_u)
            graph.remove_edge(node, v, key=key_v)
            
            # Remove the merged 'intersection' node
            graph.remove_node(node)
            
            merged = True
            break  # Exit loops to restart
        
    print("Completed merging of degree-two intersection nodes.")
    return graph

def visualize_graph(skeleton, graph, plant_name, save_path=None, show_node_types=False):
    """
    Visualize the edges of the graph with a thickness of 1 pixel. If show_node_types is True,
    also display the nodes with different colors based on their type, and show the node IDs.

    Args:
        skeleton (np.ndarray): Skeletonized image (binary).
        graph (nx.MultiGraph): Graph to visualize.
        plant_name (str): Name of the plant for the graph title.
        save_path (str, optional): Path to save the graph image. If None, return the image.
        show_node_types (bool, optional): If True, display nodes with different colors.
    """
    import cv2
    # Create an empty RGB image with the same dimensions as the skeleton
    graph_image = np.zeros((*skeleton.shape, 3), dtype=np.uint8)  # RGB image

    # Draw edges in white
    for u, v, key, data in graph.edges(keys=True, data=True):
        path = data.get('path', [])
        for y, x in path:
            if 0 <= y < graph_image.shape[0] and 0 <= x < graph_image.shape[1]:
                graph_image[int(y), int(x)] = (255, 255, 255)  # White

    if show_node_types:
        # Draw nodes with size 3x3 pixels and show node ID
        node_size = 3  # Node size
        half_size = node_size // 2

        for node_id, data in graph.nodes(data=True):
            y, x = data.get('coord')
            label = data.get('label')
            node_id_display = node_id
            if label == 'external':
                color = (255, 0, 0)  # Red
            elif label == 'intersection':
                color = (0, 255, 0)  # Green
            elif label == 'intermed':
                color = (0, 127, 127)  # Color for 'intermed'
            else:
                color = (0, 0, 255)  # Blue for other node types

            # Calculate coordinates of the rectangle around the node
            y_min = max(int(y - half_size), 0)
            y_max = min(int(y + half_size), graph_image.shape[0] - 1)
            x_min = max(int(x - half_size), 0)
            x_max = min(int(x + half_size), graph_image.shape[1] - 1)

            # Draw the rectangle on the graph
            cv2.rectangle(graph_image, (x_min, y_min), (x_max, y_max), color, thickness=-1)

            # Draw the node ID
            font_scale = 0.3
            font_thickness = 1
            text = str(node_id_display)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = x_min
            text_y = y_min - 5  # Position the text above the rectangle
            if text_y < 0:
                text_y = y_max + text_size[1] + 5  # If it goes out of image, position it below

            cv2.putText(graph_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    else:
        # Draw nodes in white (optional)
        for node_id, data in graph.nodes(data=True):
            y, x = data.get('coord')
            if 0 <= y < graph_image.shape[0] and 0 <= x < graph_image.shape[1]:
                graph_image[int(y), int(x)] = (255, 255, 255)
        # Convert the image to grayscale
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_BGR2GRAY)

    if save_path:
        io.imsave(save_path, graph_image)
        #print(f"Graph visualization saved to {save_path}")
    
    return graph_image

def get_pruned_skeleton_graph(img_name="", skeleton=None, saving_path=None):
    img_name = img_name.replace('.jpg', '')
    # flipped_skel = np.flipud(skeleton)
    flipped_skel = skeleton
    image = flipped_skel > 0.5
    skeleton = skeletonize(image > 0).astype(np.uint8)
    graph = build_graph_from_skeleton(skeleton)
    # Prune 'external' nodes with a minimum threshold of 30
    pruned_graph = prune_external_nodes(graph, min_length=30)
    # Merge 'intersection' nodes with degree 2
    merged_graph = merge_degree_two_intersections(pruned_graph)

    # Visualize the pruned and merged graph
    pruned_skeleton_img = visualize_graph(skeleton, merged_graph, plant_name=img_name)
    pruned_skeleton_img = skeletonize(pruned_skeleton_img > 0).astype(np.uint8)

    graph = build_graph_from_skeleton(pruned_skeleton_img)
        # Merge close intersection nodes
    threshold_distance = 3
    graph = merge_close_nodes(graph, threshold_distance)
    
    visualize_graph(pruned_skeleton_img, graph, plant_name=img_name, show_node_types=True, save_path=saving_path)
    
    return graph