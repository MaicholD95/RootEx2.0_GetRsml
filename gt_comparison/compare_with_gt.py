import cv2
import json
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import Path_selection.path_selection as ps
from Pwalking.walking_utils import compare_paths

alphas = [3]
betas = [4]
gammas = [0.5,0.7]
w_locals = [0.5,0.7]
w_globals = [0.5]

# alphas = [3]
# betas = [3]
# gammas = [2]
# w_locals = [0.5]
# w_globals = [0.5]
total_combinations = len(alphas) * len(betas) * len(gammas) * len(w_locals) * len(w_globals)
print(f"Total parameter combinations to evaluate: {total_combinations}")
    
def compare_with_gt(test_dataset, gt_graph_paths, final_paths_folder,index, plant_img, multiple_tips_paths, 
                    valid_final_paths,parameter_results, parameter_node_scores):
    # Read the current GT JSON file
    with open(gt_graph_paths[index], 'r') as json_file:
        data = json.load(json_file)
    # Create a dictionary to store the GT data
    gt_tip_source_paths = {}
    image_name = data['image_name']
    associated_paths = data['associated_paths']

    for idx, path in enumerate(associated_paths, start=1):
        gt_tip_source_paths[idx] = {}
        gt_tip_source_paths[idx]['tip_coord'] = path['tip_coord']
        gt_tip_source_paths[idx]['path_coords'] = path['path_coords']

    # Ensure consistent coordinate order (x, y)
    gt_tips = np.array([gt_tip_source_paths[i]['tip_coord'][::-1] for i in gt_tip_source_paths])

    # Initialize a list to store detailed match results for visualization
    results = []

    ## Comparison ##
    # Iterate through all the parameter combinations
    cnt = 0
    for alpha, beta, gamma, w_local, w_global in itertools.product(alphas, betas, gammas, w_locals, w_globals):
        param_key = (alpha, beta, gamma, w_local, w_global)
        if param_key not in parameter_results:
            parameter_results[param_key] = []
            parameter_node_scores[param_key] = []

        cnt += 1
        print(f"Processing parameter set {cnt}/{total_combinations} for image {index+1}", end='\r', flush=True)
        # Reset variables inside the parameter loop
        final_paths = []
        normalized_average_distances = []
        node_scores = []
        assigned_pred_tips = set()
        assigned_gt_tips = set()

        # Decide which paths to keep
        img = cv2.imread(plant_img.get_image_path())
        final_paths = ps.select_best_paths(
            img, plant_img.get_graph(), multiple_tips_paths,
            valid_final_paths, plant_img.get_name(), alpha, beta, gamma, w_local, w_global
        )

        tips = []
        for path in final_paths:
            tips.append(plant_img.get_graph().nodes[path[0]]['coord'])

        predicted_tips = np.array(tips)

        # Compute the distance matrix to associate predicted tips with GT tips
        if len(predicted_tips) == 0 or len(gt_tips) == 0:
            # No matches found for this param setting
            parameter_results[param_key].append(None)
            parameter_node_scores[param_key].append(None)
            continue

        distance_matrix = cdist(predicted_tips, gt_tips)

        # Generate all candidate pairs within max_range
        max_range = 150
        candidates = []
        for pred_idx in range(len(predicted_tips)):
            for gt_idx in range(len(gt_tips)):
                distance = distance_matrix[pred_idx, gt_idx]
                if distance <= max_range:
                    candidates.append((distance, pred_idx, gt_idx))

        # Sort the candidate pairs by ascending distance
        candidates.sort()

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
                # Access GT path
                gt_path = gt_tip_source_paths[gt_idx + 1]['path_coords']

                # Compare the paths
                path_difference_metrics = compare_paths(pred_path, gt_path[::-1])

                # Collect the normalized metrics
                normalized_average_distance = path_difference_metrics['normalized_average_distance']
                normalized_maximum_distance = path_difference_metrics['normalized_maximum_distance']
                combined_metric = (normalized_maximum_distance + normalized_average_distance) / 2
                normalized_average_distances.append(combined_metric)

                # Collect node matching score
                node_matching_score = path_difference_metrics['node_matching_score']
                node_scores.append(node_matching_score)

                # Store results for visualization
                results.append({
                    'pred_tip_index': pred_idx + 1,
                    'gt_tip_index': gt_idx + 1,
                    'pred_path': pred_path,
                    'gt_path': gt_path[::-1],
                    'path_difference_metrics': path_difference_metrics
                })

        # Compute the average normalized distance and node score for this image
        if normalized_average_distances:
            avg_distance = np.mean(normalized_average_distances)
        else:
            avg_distance = None  # No matches found

        if node_scores:
            avg_node_score = np.mean(node_scores)
        else:
            avg_node_score = None

        # Accumulate results for this parameter combination
        parameter_results[param_key].append(avg_distance)
        parameter_node_scores[param_key].append(avg_node_score)

        print(f"\n Finished processing image {index+1}/{len(test_dataset)}",end='\r', flush=True)


        save_results(results, final_paths_folder, plant_img,final_paths_folder)
        
    return parameter_results, parameter_node_scores,final_paths
        
    

    
    
def compute_metrics(parameter_results, parameter_node_scores):
    # After processing all images, compute the average over images for each parameter combination
    combined_metrics = []
    for param_key, distances in parameter_results.items():
        node_scores_list = parameter_node_scores[param_key]

        # Filter out None values (images where no matches were found)
        valid_pairs = [(d, n) for d, n in zip(distances, node_scores_list) if d is not None and n is not None]
        
        if valid_pairs:
            avg_distance = np.mean([vp[0] for vp in valid_pairs])
            std_distance = np.std([vp[0] for vp in valid_pairs])
            avg_node_score = np.mean([vp[1] for vp in valid_pairs])
            std_node_score = np.std([vp[1] for vp in valid_pairs])

            # Compute a final metric that balances distance and node score
            # For example: final_metric = avg_distance / (avg_node_score + 0.0001)
            # Lower final_metric is better
            final_metric = avg_distance / (avg_node_score + 0.0001)

            combined_metrics.append((final_metric, avg_distance, std_distance, avg_node_score, std_node_score, param_key))

    # Sort the parameter combinations based on the final metric
    combined_metrics.sort(key=lambda x: x[0])

    # Print the top N parameter combinations
    top_N = 5
    print(f"\nTop {top_N} parameter combinations (considering distance and node matching score):")
    for i in range(min(top_N, len(combined_metrics))):
        final_metric, avg_distance, std_distance, avg_node_score, std_node_score, param_key = combined_metrics[i]
        alpha, beta, gamma, w_local, w_global = param_key
        print(f"\nRank {i+1}:")
        print(f"Parameters: alpha={alpha}, beta={beta}, gamma={gamma}, w_local={w_local}, w_global={w_global}")
        print(f"Avg Distance Metric: {avg_distance:.6f} (std: {std_distance:.6f})")
        print(f"Avg Node Matching Score: {avg_node_score:.6f} (std: {std_node_score:.6f})")
        print(f"Final Metric (distance/node_score): {final_metric:.6f}")
        
def save_results(results, save_folder, plant_img,final_paths_folder):
    # Save comparison images
    for match in results:
        pred_idx = match['pred_tip_index'] - 1  # zero-based index
        gt_idx = match['gt_tip_index'] - 1

        pred_path = match['pred_path']
        gt_path = match['gt_path']

        pred_path_coords = np.array(pred_path)
        gt_path_coords = np.array(gt_path)

        # Read and convert the original image
        image = cv2.imread(plant_img.get_image_path())
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a folder to save the images
        image_name = plant_img.get_name()
        save_folder = os.path.join(final_paths_folder, 'compared_paths', image_name)
        os.makedirs(save_folder, exist_ok=True)

        # Create a figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Display the original image in both subplots
        axes[0].imshow(image_rgb)
        axes[0].axis('off')
        axes[1].imshow(image_rgb)
        axes[1].axis('off')

        # Extract x and y coordinates
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
        plt.subplots_adjust(top=0.85)
        save_path = os.path.join(save_folder, f'comparison_{image_name}_pair_{pred_idx + 1}.png')
        plt.savefig(save_path)
        plt.close()