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
from Pwalking.path_walking import get_all_valids_paths
from Graph.graph_processing import get_pruned_skeleton_graph, move_points_to_nearest_node
from Graph.graph_utils import divide_paths_with_equidistant_nodes
from Graph.visualization import visualize_graph, print_graph_on_original_img
import Path_selection.path_selection as ps
from Pwalking.walking_utils import compare_paths
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from gt_comparison.compare_with_gt import compare_with_gt,compute_metrics
from Rsml.create_rsml import create_RSML
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\best_models\\best_model_Exp_6_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.5_0.5_8_20.pth'
    dataset_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\test_images'
    skeletons_saving_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\skeletons'
    gt_graph_folder_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\test_images\graph_paths'
    overlapped_graphs_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\overlapped_graphs'
    final_paths_folder = r'C:\Users\maich\Desktop\rootex3\final_paths'
    rsml_output_folder = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\rsml_output'
    final_paths_imgs = []
    # Instantiate Predictor with desired thresholds and parameters
    predictor = Predictor(
        model_path,
        device=device,
        resize_height=1400,
        resize_width=1400,
        root_threshold=0.6,
        tip_threshold=0.63,
        source_threshold=0.3,
        sigma=15,
        area_threshold=320,
        circle_radius=20,
        spacing_radius=18
    )

    test_json_files = glob(os.path.join(dataset_path, '*.json'))
    test_dataset = CustomRGBDataset(json_files=test_json_files, image_dir=dataset_path, phase='test', isTraining=False)
    gt_graph_paths = glob(os.path.join(gt_graph_folder_path, '*.json'))
    print(f"Number of images in the test dataset: {len(test_dataset)}")

    # Initialize a dictionary to store results for each parameter combination
    # We'll store both the distance metric and node matching scores
    parameter_results = {}  # param_key -> list of avg_distances per image
    parameter_node_scores = {}  # param_key -> list of avg node matching scores per image

    for index in range(len(test_dataset)):
        print(f"\nProcessing image {index+1}/{len(test_dataset)}")
        #print image name
        print(test_dataset.data[index][0]['root']['name'])
        # Predict and visualize the segmentation for the current image in the test dataset
        plant_img = predictor.predict_and_visualize(test_dataset, index)

        plant_img.set_image_path(f'{dataset_path}\\{plant_img.get_name()}')

        ## Skeletonization ##
        skeleton = get_skeleton(plant_img.get_pred_mask()).astype(np.uint8) * 255
        plant_img.set_skeleton_img(skeleton)
        cv2.imwrite(f'skeleton_{plant_img.get_name()}.png', skeleton)

        ### Graph creation and processing ###
        plant_img.set_graph(get_pruned_skeleton_graph(
            plant_img.get_name(),
            skeleton,
            saving_path=f'{skeletons_saving_path}\\pruned_{plant_img.get_name()}.png'
        ))
        plant_graph_with_intermed_nodes = divide_paths_with_equidistant_nodes(
            plant_img.get_graph(), 40
        )

        plant_img.set_graph(plant_graph_with_intermed_nodes)
        visualize_graph(
            plant_img.get_skeleton_img(),
            plant_img.get_graph(),
            plant_img.get_name(),
            save_path=f'{skeletons_saving_path}\\intermed_{plant_img.get_name()}.png',
            show_node_types=True
        )
        plant_img = move_points_to_nearest_node(plant_img, 150, 100)
        visualize_graph(
            plant_img.get_skeleton_img(),
            plant_img.get_graph(),
            plant_img.get_name(),
            save_path=f'{skeletons_saving_path}\\moved_{plant_img.get_name()}.png',
            show_node_types=True
        )

        # Remove nodes and edges not connected to a source
        sources = plant_img.get_graph_sources()
        for node in list(plant_img.get_graph().nodes()):
            found = False
            for source in sources:
                if nx.has_path(plant_img.get_graph(), source, node):
                    found = True
                    break
            if not found:
                plant_img.get_graph().remove_node(node)

        visualize_graph(
            plant_img.get_skeleton_img(),
            plant_img.get_graph(),
            plant_img.get_name(),
            save_path=f'{skeletons_saving_path}\\removed_isolated_{plant_img.get_name()}.png',
            show_node_types=True
        )
        print_graph_on_original_img(plant_img, overlapped_graphs_path)

        ### Path walking ###
        # First, get all the valid paths
        multiple_tips_paths, valid_final_paths = get_all_valids_paths(plant_img)

        ## GT comparison ##
        parameter_results, parameter_node_scores,final_paths = compare_with_gt(test_dataset,gt_graph_paths,final_paths_folder,index,plant_img,multiple_tips_paths,valid_final_paths,
                        parameter_results,parameter_node_scores)
        final_paths_imgs.append(final_paths)
        
        
        #Save RSMLs
        #trasform the final_paths to the final_paths_coords
        final_paths_coords = [[] for i in range(len(final_paths))]
        
        for i in range(len(final_paths)):
            #invert coordinate from tip-source to source-tip
            final_paths[i] = final_paths[i][::-1]
            final_paths_coords[i] = [plant_img.get_graph().nodes[node]['coord'] for node in final_paths[i]]
            #final_paths_coords[i] = [plant_img.get_graph().nodes[node]['coord'] for node in final_paths[i]]
        
        
        create_RSML(plant_img.get_name(),final_paths_coords,plant_img.get_graph(),rsml_output_folder)
        print(f"RSML created for image {index+1}/{len(test_dataset)}")
        
        
    print("Results:")
    print(parameter_results)
    print(parameter_node_scores)
    compute_metrics(parameter_results,parameter_node_scores)
        
        
        
    