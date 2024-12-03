# main.py
import torch
import numpy as np
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
from Plant_info.Plant import Plant_img
from Skeleton.skeleton import get_skeleton
from Skeleton.sknw import build_sknw
from skimage.morphology import skeletonize
from skimage import io
import networkx as nx
from Pwalking.path_walking import get_all_valids_paths
from Graph.graph_processing import get_pruned_skeleton_graph, move_points_to_nearest_node
from Graph.graph_utils import divide_paths_with_equidistant_nodes
from Graph.visualization import visualize_graph, print_graph_on_original_img
import Path_selection.path_selection as ps
from Pwalking.walking_utils import compare_paths
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\best_models\\best_model_Exp_6_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.5_0.5_8_20.pth'
    dataset_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\test_images'
    skeletons_saving_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\skeletons'
    gt_graph_folder_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\test_images\graph_paths'

    plant_imgs = []
    # Instantiate Predictor with desired thresholds and parameters
    predictor = Predictor(
        model_path,
        device=device,
        resize_height=1400,
        resize_width=1400,
        root_threshold=0.5,
        tip_threshold=0.7,
        source_threshold=0.3,
        sigma=15,            # Custom sigma value
        area_threshold=320,  # Custom area_threshold
        circle_radius=20,    # Custom circle_radius for number_of_circles calculation
        spacing_radius=18    # Custom spacing_radius for spacing calculation
    )

    test_json_files = glob(os.path.join(dataset_path, '*.json'))
    test_dataset = CustomRGBDataset(json_files=test_json_files, image_dir=dataset_path, phase='test', isTraining=False)
    gt_graph_paths = glob(os.path.join(gt_graph_folder_path, '*.json'))
    print(f"Number of images in the test dataset: {len(test_dataset)}")
    avg_distances = []
    for index in range(len(test_dataset)):
        normalized_average_distances = []  # List to collect normalized average distances
        print(f"\nProcessing image {index+1}/{len(test_dataset)}")

        # Predict and visualize the segmentation for the current image in the test dataset
        plant_img = predictor.predict_and_visualize(test_dataset, index)
        plant_img.set_image_path(f'{dataset_path}\\{plant_img.get_name()}')

        ## skeletonization ##
        # Obtain the skeleton from the predicted mask and convert it to an 8-bit unsigned integer image
        skeleton = get_skeleton(plant_img.get_pred_mask()).astype(np.uint8) * 255
        # Set the skeleton image in the plant_img object for further processing or visualization
        plant_img.set_skeleton_img(skeleton)
        # Save the skeleton image with a filename based on the plant image name
        cv2.imwrite(f'skeleton_{plant_img.get_name()}.png', skeleton)

        ### Graph creation and processing ####
        # Build and prune the skeleton graph, then set it in the plant_img object
        plant_img.set_graph(get_pruned_skeleton_graph(
            plant_img.get_name(),
            skeleton,
            saving_path=f'{skeletons_saving_path}\\pruned_{plant_img.get_name()}.png'
        ))
        # Divide paths in the graph by adding equidistant intermediate nodes every 80 pixels
        plant_graph_with_intermed_nodes = divide_paths_with_equidistant_nodes(
            plant_img.get_graph(), 80
        )

        # Update the graph in the plant_img object with the new graph containing intermediate nodes
        plant_img.set_graph(plant_graph_with_intermed_nodes)
        # Visualize the updated graph overlaid on the skeleton image
        visualize_graph(
            plant_img.get_skeleton_img(),
            plant_img.get_graph(),
            plant_img.get_name(),
            save_path=f'{skeletons_saving_path}\\intermed_{plant_img.get_name()}.png',
            show_node_types=True
        )
        # Move the sources and tips to the nearest node in the graph within a distance threshold of 60 pixels
        plant_img = move_points_to_nearest_node(plant_img, 60, 100)
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
        # Visualize the updated graph overlaid on the skeleton image
        visualize_graph(
            plant_img.get_skeleton_img(),
            plant_img.get_graph(),
            plant_img.get_name(),
            save_path=f'{skeletons_saving_path}\\removed_isolated_{plant_img.get_name()}.png',
            show_node_types=True
        )
        print_graph_on_original_img(plant_img, r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\overlapped_graphs')

        ### path walking ###
        final_paths = []
        multiple_tips_paths = []
        final_paths = []
        # First, get all the valid paths
        multiple_tips_paths, final_paths = get_all_valids_paths(plant_img)
        # Decide which paths to keep
        img = cv2.imread(plant_img.get_image_path())
        final_paths = ps.select_best_paths(img, plant_img.get_graph(), multiple_tips_paths, final_paths, plant_img.get_name())

        # Read the current GT JSON file
        with open(gt_graph_paths[index], 'r') as json_file:
            data = json.load(json_file)
        # Create a dictionary to store the GT data
        gt_tip_source_paths = {}

        image_name = data['image_name']
        associated_paths = data['associated_paths']

        # Iterate through the associated paths
        for idx, path in enumerate(associated_paths, start=1):
            gt_tip_source_paths[idx] = {}
            gt_tip_source_paths[idx]['tip_coord'] = path['tip_coord']
            gt_tip_source_paths[idx]['path_coords'] = path['path_coords']

        # Get all tips from final paths
        tips = []
        for path in final_paths:
            tips.append(plant_img.get_graph().nodes[path[0]]['coord'])

        predicted_tips = np.array(tips)  # List of predicted tip coordinates

        # Ensure consistent coordinate order (x, y)
        gt_tips = np.array([gt_tip_source_paths[i]['tip_coord'][::-1] for i in gt_tip_source_paths])  # GT tip coordinates

        print('len of predicted tips:', len(predicted_tips))
        print('len of gt tips:', len(gt_tips))
        # Compute the distance matrix
        distance_matrix = cdist(predicted_tips, gt_tips)

        # Generate all candidate pairs within max_range
        max_range = 300  # Define your maximum allowed distance
        candidates = []
        for pred_idx in range(len(predicted_tips)):
            for gt_idx in range(len(gt_tips)):
                distance = distance_matrix[pred_idx, gt_idx]
                if distance <= max_range:
                    candidates.append((distance, pred_idx, gt_idx))

        # Sort the candidate pairs by ascending distance
        candidates.sort()

        # Initialize sets to keep track of assigned tips
        assigned_pred_tips = set()
        assigned_gt_tips = set()

        # List to store the results
        results = []

        # Iterate over the sorted candidate pairs
        for distance, pred_idx, gt_idx in candidates:
            if pred_idx not in assigned_pred_tips and gt_idx not in assigned_gt_tips:
                # Assign the match
                assigned_pred_tips.add(pred_idx)
                assigned_gt_tips.add(gt_idx)

                # Retrieve necessary information
                pred_tip = predicted_tips[pred_idx]
                gt_tip = gt_tips[gt_idx]
                pred_path = [
                    plant_img.get_graph().nodes[node]['coord'] for node in final_paths[pred_idx]
                ]
                # Access GT path; ensure correct indexing
                gt_path = gt_tip_source_paths[gt_idx + 1]['path_coords']  # +1 if GT indices start from 1

                # Compare the paths
                path_difference_metrics = compare_paths(pred_path, gt_path[::-1])  # Reverse GT path for correct comparison
                print(f"Path difference metrics: {path_difference_metrics}")

                # Append to results
                results.append({
                    'pred_tip_index': pred_idx + 1,
                    'gt_tip_index': gt_idx + 1,
                    'path_difference_metrics': path_difference_metrics,
                    'pred_path': pred_path,
                    'gt_path': gt_path,
                })

                # Collect the normalized average distance for this path
                normalized_average_distance = path_difference_metrics['normalized_average_distance']
                normalized_average_distances.append(normalized_average_distance)

        # Compute the average normalized average distance for this image
        if normalized_average_distances:
            avg_distance = np.mean(normalized_average_distances)
            avg_distances.append(avg_distance)
        else:
            avg_distances.append(None)  # No matches found

        # Plotting and saving the images
        for match in results:
            pred_idx = match['pred_tip_index'] - 1  # Adjusting for zero-based index
            gt_idx = match['gt_tip_index'] - 1

            pred_path = match['pred_path']
            gt_path = match['gt_path']

            pred_path_coords = np.array(pred_path)
            gt_path_coords = np.array(gt_path)

            # Read and convert the original image
            image = cv2.imread(plant_img.get_image_path())
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create a folder to save the images
            image_name = plant_img.get_name()  # Replace 'img' with the image name
            save_folder = os.path.join('final_paths', 'compared_paths', image_name)

            # Create the directory if it doesn't exist
            os.makedirs(save_folder, exist_ok=True)

            # Create a figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Display the original image in both subplots
            axes[0].imshow(image_rgb)
            axes[0].axis('off')

            axes[1].imshow(image_rgb)
            axes[1].axis('off')

            # Extract x and y coordinates (ensure correct coordinate order)
            # Swap x and y if necessary
            gt_x = gt_path_coords[:, 1]
            gt_y = gt_path_coords[:, 0]

            pred_x = pred_path_coords[:, 1]
            pred_y = pred_path_coords[:, 0]

            # Plot the GT path on the left subplot
            axes[0].plot(gt_x, gt_y, color='red', linewidth=2)

            # Plot the predicted path on the right subplot
            axes[1].plot(pred_x, pred_y, color='blue', linewidth=2)

            # Retrieve the metrics for this path
            path_difference_metrics = match['path_difference_metrics']
            normalized_average_distance = path_difference_metrics['normalized_average_distance']
            normalized_maximum_distance = path_difference_metrics['normalized_maximum_distance']
            average_angular_difference = path_difference_metrics['average_angular_difference']
            node_matching_score = path_difference_metrics['node_matching_score']

            # Format the metrics into strings for titles
            metrics_text_left = (
                f"Ground Truth Path\n"
                f"Node Matching Score: {node_matching_score:.2f}"
            )
            metrics_text_right = (
                f"Predicted Path\n"
                f"Norm Avg Dist: {normalized_average_distance:.4f}\n"
                f"Norm Max Dist: {normalized_maximum_distance:.4f}\n"
                f"Avg Angular Diff: {average_angular_difference:.4f} rad"
            )

            # Set the titles with metrics
            axes[0].set_title(metrics_text_left, fontsize=10)
            axes[1].set_title(metrics_text_right, fontsize=10)

            # Adjust layout and save the figure
            plt.tight_layout()
            # If titles are being cut off, adjust the top margin
            plt.subplots_adjust(top=0.85)
            save_path = os.path.join(save_folder, f'comparison_{image_name}_pair_{pred_idx + 1}.png')
            plt.savefig(save_path)
            plt.close()

        print('img:', image_name)
        print('done')

    print('Average distances:', avg_distances)
