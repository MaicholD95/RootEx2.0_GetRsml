# main.py
import os
import sys
import random
from glob import glob
from collections import defaultdict
import concurrent.futures

import cv2
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # safe for headless environments

# ---------------- Reproducibility ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- Project imports ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dataset import CustomRGBDataset
from post_process_predicted_mask import Predictor
from Skeleton.skeleton import get_skeleton
from Graph.graph_processing import get_pruned_skeleton_graph, move_points_to_nearest_node
from Graph.graph_utils import divide_paths_with_equidistant_nodes
from Graph.visualization import visualize_graph, print_graph_on_original_img
from Pwalking.path_walking_new import get_all_valids_paths
from gt_comparison.compare_with_gt import compare_with_gt, compute_metrics
from Rsml.create_rsml import create_RSML


def _label_extra_tips(plant_img, min_chain_len=4):
    """
    Heuristic: if a leaf (degree=1) is not labeled as tip, but lies after a
    simple chain of length >= min_chain_len, mark it as a tip.
    """
    G = plant_img.get_graph()
    tips = set(plant_img.get_graph_tips())

    for node in list(G.nodes()):
        if G.degree[node] == 1 and node not in tips:
            current = node
            visited = {current}
            depth = 0
            while True:
                neighbors = list(G.neighbors(current))
                next_nodes = [n for n in neighbors if n not in visited]
                if len(next_nodes) != 1:
                    break  # not a simple path
                current = next_nodes[0]
                visited.add(current)
                depth += 1
                if depth >= min_chain_len:
                    tips.add(node)
                    plant_img.modify_label(node, 'tip')
                    break


def _remove_isolated_from_sources(plant_img):
    """
    Remove nodes that are not connected to any source node.
    """
    G = plant_img.get_graph()
    sources = plant_img.get_graph_sources()
    for node in list(G.nodes()):
        if not any(nx.has_path(G, s, node) for s in sources):
            G.remove_node(node)


def process_image(index, test_dataset, predictor, gt_graph_paths, dataset_path,
                  skeletons_saving_path, overlapped_graphs_path, final_paths_folder,
                  rsml_output_folder):
    """
    Full per-image pipeline:
    1) Predict masks and build graph.
    2) Prune/enrich graph, move anchors, minor relabeling.
    3) Collect candidate paths.
    4) If GT exists → evaluate grid and compute metrics; else → pick default params (no metrics).
    5) Save visualizations and write RSML from final paths.
    """
    parameter_results = defaultdict(list)
    parameter_node_scores = defaultdict(list)

    # Predict and prepare the PlantImage object
    plant_img = predictor.predict_and_visualize(test_dataset, index)
    plant_img.set_image_path(os.path.join(dataset_path, plant_img.get_name()))

    # Skeleton and initial pruned graph
    skeleton = get_skeleton(plant_img.get_pred_mask()).astype(np.uint8) * 255
    plant_img.set_skeleton_img(skeleton)
    os.makedirs(skeletons_saving_path, exist_ok=True)
    cv2.imwrite(os.path.join(skeletons_saving_path, f'skeleton_{plant_img.get_name()}.png'), skeleton)

    plant_img.set_graph(get_pruned_skeleton_graph(
        plant_img.get_name(), skeleton,
        saving_path=os.path.join(skeletons_saving_path, f'pruned_{plant_img.get_name()}.png'))
    )

    # Insert intermediate nodes to smooth distances along edges
    graph_with_intermediate = divide_paths_with_equidistant_nodes(plant_img.get_graph(), 40)
    plant_img.set_graph(graph_with_intermediate)

    visualize_graph(
        plant_img.get_skeleton_img(), plant_img.get_graph(), plant_img.get_name(),
        save_path=os.path.join(skeletons_saving_path, f'intermediate_{plant_img.get_name()}.png'),
        show_node_types=True
    )

    # Move tips/sources to nearest nodes to stabilize anchors
    plant_img = move_points_to_nearest_node(plant_img, 150, 100)

    # Heuristic: mark more tips at long leaf chains
    _label_extra_tips(plant_img, min_chain_len=4)

    visualize_graph(
        plant_img.get_skeleton_img(), plant_img.get_graph(), plant_img.get_name(),
        save_path=os.path.join(skeletons_saving_path, f'moved_{plant_img.get_name()}.png'),
        show_node_types=True
    )

    # Remove nodes not connected to any source
    _remove_isolated_from_sources(plant_img)

    visualize_graph(
        plant_img.get_skeleton_img(), plant_img.get_graph(), plant_img.get_name(),
        save_path=os.path.join(skeletons_saving_path, f'removed_isolated_{plant_img.get_name()}.png'),
        show_node_types=True
    )

    # Graph overlay on original image for quick inspection
    print_graph_on_original_img(plant_img, overlapped_graphs_path)

    # Candidate paths (per-tip enumerations and pruned valid ones)
    multiple_tips_paths, valid_final_paths = get_all_valids_paths(plant_img)

    # Compare with GT if available; otherwise select once with default params (inside compare_with_gt)
    parameter_results, parameter_node_scores, final_paths = compare_with_gt(
        test_dataset, gt_graph_paths, final_paths_folder, index, plant_img,
        multiple_tips_paths, valid_final_paths, parameter_results, parameter_node_scores
    )

    # Convert selected node indices to coords (reverse to go tip→source if your RSML expects that)
    final_paths_coords = [
        [plant_img.get_graph().nodes[node]['coord'] for node in path[::-1]]
        for path in final_paths
    ]
    create_RSML(plant_img.get_name(), final_paths_coords, plant_img.get_graph(), rsml_output_folder)
    print(f"RSML created for image {index+1}/{len(test_dataset)}")

    return parameter_results, parameter_node_scores


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- Paths / Config ----------
    model_path             = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\best_models\best_model_Exp_6_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.5_0.5_8_20.pth'
    dataset_path           = r'D:\WORK\DATA\root\failure_imgs'
    skeletons_saving_path  = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\skeleton'
    gt_graph_folder_path   = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\test_images\graph_paths'
    overlapped_graphs_path = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\overlapped_graphs'
    final_paths_folder     = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\final_paths'
    rsml_output_folder     = r'D:\WORK\DATA\root\failure_imgs\re2'

    os.makedirs(overlapped_graphs_path, exist_ok=True)
    os.makedirs(final_paths_folder, exist_ok=True)
    os.makedirs(rsml_output_folder, exist_ok=True)

    # ---------- Predictor ----------
    predictor = Predictor(
        model_path, device=device,
        resize_height=1400, resize_width=1200,
        root_threshold=0.45, tip_threshold=0.5,
        source_threshold=0.53, sigma=15,
        area_threshold=320, circle_radius=20, spacing_radius=18
    )

    # ---------- Dataset / GT list ----------
    test_json_files = glob(os.path.join(dataset_path, '*.json'))
    test_dataset = CustomRGBDataset(test_json_files, dataset_path, phase='test', isTraining=False)

    # If empty/missing, GT-optional mode will kick in inside compare_with_gt
    gt_graph_paths = glob(os.path.join(gt_graph_folder_path, '*.json')) if os.path.isdir(gt_graph_folder_path) else []

    print(f"Number of images in the test dataset: {len(test_dataset)}")
    if not gt_graph_paths:
        print("GT folder missing or empty — running in GT-optional mode (metrics will be skipped).")

    # ---------- Parallel processing ----------
    global_parameter_results     = defaultdict(list)
    global_parameter_node_scores = defaultdict(list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(
                process_image, i, test_dataset, predictor, gt_graph_paths,
                dataset_path, skeletons_saving_path, overlapped_graphs_path,
                final_paths_folder, rsml_output_folder
            ) for i in range(len(test_dataset))
        ]

        for f in concurrent.futures.as_completed(futures):
            res_results, res_scores = f.result()
            for k, v in res_results.items():
                global_parameter_results[k].extend(v)
            for k, v in res_scores.items():
                global_parameter_node_scores[k].extend(v)

    # ---------- Metrics (only meaningful if GT was present) ----------
    print("\nResults (raw dicts):")
    print(global_parameter_results)
    print(global_parameter_node_scores)
    compute_metrics(global_parameter_results, global_parameter_node_scores)
