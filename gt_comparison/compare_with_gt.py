# gt_comparison/compare_with_gt.py
"""
GT-optional comparison and path selection utilities.

Behavior:
- If GT for the current image is available and valid:
    * Evaluate all parameter combinations.
    * Match predicted tips to GT tips (within a radius).
    * Compute path difference metrics and aggregate results.
    * Optionally save side-by-side comparison figures.

- If GT is missing/invalid:
    * Select final paths once using the first tuple from (alphas, betas, gammas, w_locals, w_globals).
    * Save a simple "predicted-only" figure.
    * Skip all metrics.

Exposed functions:
- compare_with_gt(...)
- compute_metrics(...)
"""

import os
import json
import itertools
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import Path_selection.path_selection as ps
from Pwalking.walking_utils import compare_paths

# Parameter grid (the first value of each list is used in GT-missing mode)
alphas    = [5]
betas     = [15]
gammas    = [0.2]
w_locals  = [0.4]
w_globals = [0.6]

total_combinations = len(alphas) * len(betas) * len(gammas) * len(w_locals) * len(w_globals)
print(f"Total parameter combinations to evaluate: {total_combinations}")


# ----------------------------- Internal helpers ----------------------------- #

def _gt_available(gt_graph_paths, index):
    """
    Return True if a GT JSON is available and parseable for the given index.
    Required structure:
        data['associated_paths'] : list of dicts with keys 'tip_coord' and 'path_coords'
    """
    if not gt_graph_paths:
        return False
    if index < 0 or index >= len(gt_graph_paths):
        return False

    gt_path = gt_graph_paths[index]
    if not os.path.isfile(gt_path):
        return False

    try:
        with open(gt_path, "r") as f:
            data = json.load(f)
        aps = data.get("associated_paths", None)
        if not isinstance(aps, list) or len(aps) == 0:
            return False
        sample = aps[0]
        if "tip_coord" not in sample or "path_coords" not in sample:
            return False
    except Exception:
        return False

    return True


def _select_paths_single_setting(plant_img, multiple_tips_paths, valid_final_paths,
                                 alpha, beta, gamma, w_local, w_global):
    """
    Run path selection once with a specific parameter tuple.
    """
    img = cv2.imread(plant_img.get_image_path())
    final_paths = ps.select_best_paths(
        img, plant_img.get_graph(), multiple_tips_paths,
        valid_final_paths, plant_img.get_name(), alpha, beta, gamma, w_local, w_global
    )
    return final_paths


def _save_predicted_only(plant_img, final_paths, save_root):
    """
    Save a single-figure visualization of predicted paths overlaid on the original image.
    Used in GT-missing mode.
    """
    os.makedirs(save_root, exist_ok=True)

    image = cv2.imread(plant_img.get_image_path())
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_rgb)
    ax.axis("off")
    ax.set_title("Predicted paths (no GT available)")

    # Draw each predicted path
    for path in final_paths:
        coords = np.array([plant_img.get_graph().nodes[n]["coord"] for n in path])
        xs = coords[:, 1]  # col
        ys = coords[:, 0]  # row
        ax.plot(xs, ys, linewidth=2)

    out_path = os.path.join(save_root, f"predicted_only_{plant_img.get_name()}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _save_results_pairs(results, final_paths_folder, plant_img):
    """
    Save side-by-side figures for matched GT/pred pairs with metrics in titles.
    """
    for match in results:
        pred_idx = match['pred_tip_index'] - 1
        gt_idx   = match['gt_tip_index'] - 1

        pred_path = np.array(match['pred_path'])
        gt_path   = np.array(match['gt_path'])

        image = cv2.imread(plant_img.get_image_path())
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_name = plant_img.get_name()
        save_folder = os.path.join(final_paths_folder, 'compared_paths', image_name)
        os.makedirs(save_folder, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_rgb); axes[0].axis('off')
        axes[1].imshow(image_rgb); axes[1].axis('off')

        # coords are (row, col) -> x=col, y=row
        gt_x, gt_y = gt_path[:, 1], gt_path[:, 0]
        pr_x, pr_y = pred_path[:, 1], pred_path[:, 0]

        axes[0].plot(gt_x, gt_y, linewidth=2)
        axes[1].plot(pr_x, pr_y, linewidth=2)

        m = match['path_difference_metrics']
        left_title  = f"Ground Truth Path\nNode Matching Score: {m['node_matching_score']:.2f}"
        right_title = (
            f"Predicted Path\n"
            f"Norm Avg Dist: {m['normalized_average_distance']:.4f}\n"
            f"Norm Max Dist: {m['normalized_maximum_distance']:.4f}\n"
            f"Avg Angular Diff: {m['average_angular_difference']:.4f} rad"
        )
        axes[0].set_title(left_title,  fontsize=10)
        axes[1].set_title(right_title, fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        save_path = os.path.join(save_folder, f'comparison_{image_name}_pair_{pred_idx + 1}.png')
        plt.savefig(save_path)
        plt.close()


# --------------------------------- Public API -------------------------------- #

def compare_with_gt(test_dataset, gt_graph_paths, final_paths_folder, index, plant_img,
                    multiple_tips_paths, valid_final_paths, parameter_results, parameter_node_scores):
    """
    If GT exists for the current image, evaluate path selection across the full parameter grid
    and compute metrics. If GT is not available, select paths once using the first tuple of
    (alpha, beta, gamma, w_local, w_global) and skip all metrics/comparisons.

    Returns:
        parameter_results, parameter_node_scores, final_paths
    """
    # -------------------- GT missing/invalid: single-shot selection -------------------- #
    if not _gt_available(gt_graph_paths, index):
        a, b, g, wl, wg = alphas[0], betas[0], gammas[0], w_locals[0], w_globals[0]
        final_paths = _select_paths_single_setting(
            plant_img, multiple_tips_paths, valid_final_paths, a, b, g, wl, wg
        )
        _save_predicted_only(plant_img, final_paths, os.path.join(final_paths_folder, "predicted_only"))
        return parameter_results, parameter_node_scores, final_paths

    # -------------------- GT available: normal evaluation loop -------------------- #
    with open(gt_graph_paths[index], 'r') as json_file:
        data = json.load(json_file)

    gt_tip_source_paths = {}
    image_name = data.get('image_name', plant_img.get_name())
    associated_paths = data['associated_paths']

    for idx, path in enumerate(associated_paths, start=1):
        gt_tip_source_paths[idx] = {
            'tip_coord': path['tip_coord'],
            'path_coords': path['path_coords']
        }

    # GT tips for association (flip to (x, y) from (y, x))
    gt_tips = np.array([gt_tip_source_paths[i]['tip_coord'][::-1] for i in gt_tip_source_paths])

    cnt = 0
    last_final_paths = None  # returned as "final" paths for RSML

    for alpha, beta, gamma, w_local, w_global in itertools.product(alphas, betas, gammas, w_locals, w_globals):
        param_key = (alpha, beta, gamma, w_local, w_global)
        if param_key not in parameter_results:
            parameter_results[param_key] = []
            parameter_node_scores[param_key] = []

        cnt += 1
        print(f"Processing parameter set {cnt}/{total_combinations} for image {index+1}", end='\r', flush=True)

        # Path selection for current parameter set
        img = cv2.imread(plant_img.get_image_path())
        final_paths = ps.select_best_paths(
            img, plant_img.get_graph(), multiple_tips_paths,
            valid_final_paths, plant_img.get_name(), alpha, beta, gamma, w_local, w_global
        )
        last_final_paths = final_paths

        # Predicted tips from selected paths
        tips = [plant_img.get_graph().nodes[path[0]]['coord'] for path in final_paths]
        predicted_tips = np.array(tips)

        # If either side is empty, record None and continue
        if len(predicted_tips) == 0 or len(gt_tips) == 0:
            parameter_results[param_key].append(None)
            parameter_node_scores[param_key].append(None)
            continue

        # Associate predicted tips to GT tips by distance
        distance_matrix = cdist(predicted_tips, gt_tips)
        max_range = 150
        candidates = [
            (distance_matrix[p, g], p, g)
            for p in range(len(predicted_tips))
            for g in range(len(gt_tips))
            if distance_matrix[p, g] <= max_range
        ]
        candidates.sort()

        assigned_pred = set()
        assigned_gt = set()
        normalized_average_distances = []
        node_scores = []
        results_for_fig = []

        for distance, pred_idx, gt_idx in candidates:
            if pred_idx in assigned_pred or gt_idx in assigned_gt:
                continue
            assigned_pred.add(pred_idx)
            assigned_gt.add(gt_idx)

            pred_path = [plant_img.get_graph().nodes[node]['coord'] for node in final_paths[pred_idx]]
            gt_path = gt_tip_source_paths[gt_idx + 1]['path_coords']

            # Compare paths (reverse GT path if needed to match your original convention)
            metrics = compare_paths(pred_path, gt_path[::-1])

            # Combine normalized distances (lower is better)
            nad = metrics['normalized_average_distance']
            nmd = metrics['normalized_maximum_distance']
            combined = (nmd + nad) / 2.0
            normalized_average_distances.append(combined)

            # Node matching score (higher is better)
            node_scores.append(metrics['node_matching_score'])

            # Keep info for optional side-by-side figure
            results_for_fig.append({
                'pred_tip_index': pred_idx + 1,
                'gt_tip_index': gt_idx + 1,
                'pred_path': pred_path,
                'gt_path': gt_path[::-1],
                'path_difference_metrics': metrics
            })

        # Aggregate per-image numbers for this parameter set
        avg_distance = float(np.mean(normalized_average_distances)) if normalized_average_distances else None
        avg_node_score = float(np.mean(node_scores)) if node_scores else None

        parameter_results[param_key].append(avg_distance)
        parameter_node_scores[param_key].append(avg_node_score)

        # Optional: save side-by-side figures
        if results_for_fig:
            _save_results_pairs(results_for_fig, final_paths_folder, plant_img)

    return parameter_results, parameter_node_scores, (last_final_paths or [])


def compute_metrics(parameter_results, parameter_node_scores):
    """
    Compute aggregate metrics across images for each parameter combination.
    If nothing valid is found (e.g., no GT), print a notice and exit.
    """
    combined_metrics = []

    for param_key, distances in parameter_results.items():
        node_scores_list = parameter_node_scores.get(param_key, [])
        valid_pairs = [(d, n) for d, n in zip(distances, node_scores_list) if d is not None and n is not None]

        if valid_pairs:
            avg_distance   = float(np.mean([vp[0] for vp in valid_pairs]))
            std_distance   = float(np.std([vp[0] for vp in valid_pairs]))
            avg_node_score = float(np.mean([vp[1] for vp in valid_pairs]))
            std_node_score = float(np.std([vp[1] for vp in valid_pairs]))
            final_metric   = avg_distance / (avg_node_score + 1e-4)  # lower is better

            combined_metrics.append((final_metric, avg_distance, std_distance, avg_node_score, std_node_score, param_key))

    if not combined_metrics:
        print("\nNo GT-based metrics to report (GT missing or no valid matches).")
        return

    combined_metrics.sort(key=lambda x: x[0])
    top_N = 5
    print(f"\nTop {top_N} parameter combinations (distance/node_score):")
    for i in range(min(top_N, len(combined_metrics))):
        fm, avg_d, std_d, avg_n, std_n, pkey = combined_metrics[i]
        a, b, g, wl, wg = pkey
        print(f"\nRank {i+1}:")
        print(f"Parameters: alpha={a}, beta={b}, gamma={g}, w_local={wl}, w_global={wg}")
        print(f"Avg Distance Metric: {avg_d:.6f} (std: {std_d:.6f})")
        print(f"Avg Node Matching Score: {avg_n:.6f} (std: {std_n:.6f})")
        print(f"Final Metric (distance/node_score): {fm:.6f}")
