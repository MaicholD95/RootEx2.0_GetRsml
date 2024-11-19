# graph_processing.py

from Graph.graph_utils import (
    build_graph_from_skeleton,
    prune_external_nodes,
    merge_degree_two_intersections,
    merge_close_nodes,
    get_nearest_node_in_range
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


def move_points_to_nearest_node(plant_img, s_distance_threshold=5, t_distance_threshold=5):

    def move_sources():
        for i,source in enumerate(plant_img.get_sources()):
            nearest_node_source_id = get_nearest_node_in_range(plant_img.get_graph(), source,radius_min=0, radius_max=s_distance_threshold)
            if nearest_node_source_id != -1:
                new_sources = plant_img.get_sources()
                new_sources[i] = plant_img.get_graph().nodes[nearest_node_source_id]['coord']
                plant_img.set_sources(new_sources)
                plant_img.modify_label(nearest_node_source_id,'source')
            else:
                print(f"Source {source} not found in the graph")
            
    def move_tips():
        for i,tip in enumerate(plant_img.get_tips()):
            nearest_node_tip_id = get_nearest_node_in_range(plant_img.get_graph(), tip,radius_min=0, radius_max=t_distance_threshold)
            if nearest_node_tip_id != -1:
                new_tips = plant_img.get_tips()
                new_tips[i] = plant_img.get_graph().nodes[nearest_node_tip_id]['coord']
                plant_img.set_tips(new_tips)
                plant_img.modify_label(nearest_node_tip_id,'tip')
            else:
                print(f"Tip {tip} not found in the graph")


    move_sources()
    move_tips()
    return plant_img
    