# graph_processing.py

from Graph.graph_utils import (
    build_graph_from_skeleton,
    prune_external_nodes,
    merge_degree_two_intersections,
    merge_close_nodes,
)
from Graph.visualization import visualize_graph
from Graph.skeleton_utils import get_skeleton

def get_pruned_skeleton_graph(img_name="", skeleton=None, saving_path=None):
    img_name = img_name.replace('.jpg', '')
    image = skeleton > 0.5
    skeleton = get_skeleton(image)
    graph = build_graph_from_skeleton(skeleton)
    # Prune 'external' nodes with a minimum threshold of 30
    pruned_graph = prune_external_nodes(graph, min_length=30)
    # Merge 'intersection' nodes with degree 2
    merged_graph = merge_degree_two_intersections(pruned_graph)
    # Visualize the pruned and merged graph
    pruned_skeleton_img = visualize_graph(skeleton, merged_graph, plant_name=img_name)
    pruned_skeleton_img = get_skeleton(pruned_skeleton_img)
    # Build the graph again from the pruned skeleton image
    graph = build_graph_from_skeleton(pruned_skeleton_img)
    # Merge close intersection nodes
    threshold_distance = 3
    graph = merge_close_nodes(graph, threshold_distance)
    # Visualize the final graph
    visualize_graph(pruned_skeleton_img, graph, plant_name=img_name, show_node_types=True, save_path=saving_path)
    return graph
