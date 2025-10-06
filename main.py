# main.py
import torch
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If using multi-GPU.

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import os
import cv2
import torch
import random
import numpy as np
import networkx as nx
import concurrent.futures
from glob import glob
from collections import defaultdict
import cv2
import os
from glob import glob
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from dataset import CustomRGBDataset
from post_process_predicted_mask import Predictor
from Skeleton.skeleton import get_skeleton
import networkx as nx
#from Pwalking.path_walking_new import extract_valid_paths
from Graph.graph_processing import get_pruned_skeleton_graph, move_points_to_nearest_node
from Graph.graph_utils import divide_paths_with_equidistant_nodes
from Graph.visualization import visualize_graph, print_graph_on_original_img
import Path_selection.path_selection as ps
from Pwalking.walking_utils import compare_paths
from Pwalking.path_walking_new import get_all_valids_paths
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from gt_comparison.compare_with_gt import compare_with_gt,compute_metrics
from Rsml.create_rsml import create_RSML


def process_image(index, test_dataset, predictor, gt_graph_paths, dataset_path,
                  skeletons_saving_path, overlapped_graphs_path, final_paths_folder,
                  rsml_output_folder):

    parameter_results = defaultdict(list)
    parameter_node_scores = defaultdict(list)
    final_paths_imgs = []

    plant_img = predictor.predict_and_visualize(test_dataset, index)
    plant_img.set_image_path(os.path.join(dataset_path, plant_img.get_name()))

    skeleton = get_skeleton(plant_img.get_pred_mask()).astype(np.uint8) * 255
    plant_img.set_skeleton_img(skeleton)
    cv2.imwrite(f'skeleton_{plant_img.get_name()}.png', skeleton)

    plant_img.set_graph(get_pruned_skeleton_graph(
        plant_img.get_name(), skeleton,
        saving_path=os.path.join(skeletons_saving_path, f'pruned_{plant_img.get_name()}.png')))

    plant_graph_with_intermed_nodes = divide_paths_with_equidistant_nodes(plant_img.get_graph(), 40)
    plant_img.set_graph(plant_graph_with_intermed_nodes)

    visualize_graph(
        plant_img.get_skeleton_img(), plant_img.get_graph(), plant_img.get_name(),
        save_path=os.path.join(skeletons_saving_path, f'intermed_{plant_img.get_name()}.png'), show_node_types=True)

    plant_img = move_points_to_nearest_node(plant_img, 150, 100)
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
                    break  # Non è un cammino lineare
                current = next_nodes[0]
                visited.add(current)
                depth += 1
                if depth >= 4:
                    tips.add(node)
                    plant_img.modify_label(node, 'tip')  # <-- etichetta come tip
                    break

    visualize_graph(
        plant_img.get_skeleton_img(), plant_img.get_graph(), plant_img.get_name(),
        save_path=os.path.join(skeletons_saving_path, f'moved_{plant_img.get_name()}.png'), show_node_types=True)

    sources = plant_img.get_graph_sources()
    for node in list(plant_img.get_graph().nodes()):
        if not any(nx.has_path(plant_img.get_graph(), source, node) for source in sources):
            plant_img.get_graph().remove_node(node)

    visualize_graph(
        plant_img.get_skeleton_img(), plant_img.get_graph(), plant_img.get_name(),
        save_path=os.path.join(skeletons_saving_path, f'removed_isolated_{plant_img.get_name()}.png'), show_node_types=True)

    print_graph_on_original_img(plant_img, overlapped_graphs_path)

    multiple_tips_paths, valid_final_paths = get_all_valids_paths(plant_img)#extract_valid_paths(plant_img,45)

    parameter_results, parameter_node_scores, final_paths = compare_with_gt(
        test_dataset, gt_graph_paths, final_paths_folder, index, plant_img,
        multiple_tips_paths, valid_final_paths,
        parameter_results, parameter_node_scores)

    final_paths_imgs.append(final_paths)

    final_paths_coords = [
        [plant_img.get_graph().nodes[node]['coord'] for node in path[::-1]]
        for path in final_paths
    ]
    create_RSML(plant_img.get_name(), final_paths_coords, plant_img.get_graph(), rsml_output_folder)
    print(f"RSML created for image {index+1}/{len(test_dataset)}")

    return parameter_results, parameter_node_scores

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\best_models\\best_model_Exp_6_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.5_0.5_8_20.pth'
    dataset_path = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\test_images'
    skeletons_saving_path = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\skeletons\failure_analysis'
    gt_graph_folder_path = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\test_images\graph_paths'
    overlapped_graphs_path = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\overlapped_graphs'
    final_paths_folder = r'D:\WORK\DATA\root\RootEx3.0_GetRsml\final_paths'
    rsml_output_folder = r'D:\WORK\DATA\root\failure_imgs\re2'
    final_paths_imgs = []
    # Instantiate Predictor with desired thresholds and parameters
    predictor = Predictor(
            model_path, device=device,
            resize_height=1400, resize_width=1200,
            root_threshold=0.45, tip_threshold=0.5,
            source_threshold=0.53, sigma=15,
            area_threshold=320, circle_radius=20, spacing_radius=18)

    test_json_files = glob(os.path.join(dataset_path, '*.json'))
    test_dataset = CustomRGBDataset(test_json_files, dataset_path, phase='test', isTraining=False)
    gt_graph_paths = glob(os.path.join(gt_graph_folder_path, '*.json'))

    print(f"Number of images in the test dataset: {len(test_dataset)}")

    global_parameter_results = defaultdict(list)
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

    print("\nResults:")
    print(global_parameter_results)
    print(global_parameter_node_scores)
    compute_metrics(global_parameter_results, global_parameter_node_scores)


            
        
        
    