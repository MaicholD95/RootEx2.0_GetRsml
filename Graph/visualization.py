# visualization.py

import cv2
import numpy as np
from skimage import io

def visualize_graph(skeleton, graph, plant_name, save_path=None, show_node_types=False):
    """
    Visualize the edges of the graph with a thickness of 1 pixel. If show_node_types is True,
    also display the nodes with different colors based on their type, and show the node IDs.
    """
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
                color = (0, 127, 127)  # Teal
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
        # Draw nodes in white
        for node_id, data in graph.nodes(data=True):
            y, x = data.get('coord')
            if 0 <= y < graph_image.shape[0] and 0 <= x < graph_image.shape[1]:
                graph_image[int(y), int(x)] = (255, 255, 255)
        # Convert the image to grayscale
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_BGR2GRAY)

    if save_path:
        io.imsave(save_path, graph_image)
        print(f"Graph visualization saved to {save_path}")
    
    return graph_image
