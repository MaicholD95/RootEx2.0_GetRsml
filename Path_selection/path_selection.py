import Path_selection.path_selection_utils as psu
import random
import numpy as np
from collections import defaultdict
import math

# Definizione dell'angolo massimo consentito
# MAX_ALLOWED_ANGLE_DEG = 45  # Angolo massimo in gradi
# MAX_ALLOWED_ANGLE_RAD = np.deg2rad(MAX_ALLOWED_ANGLE_DEG)

def compute_iou(path_nodes, existing_nodes):
    intersection = path_nodes & existing_nodes
    union = path_nodes | existing_nodes
    return len(intersection) / len(union) if union else 0.0

def get_intersection_nodes(G, path):
    intersection_nodes = []
    for node in path:
        if G.nodes[node].get('label') == 'intersection':
            intersection_nodes.append(node)
    return intersection_nodes

def compute_global_angle_deviation(G, path_coords, path_nodes):
    tip_coord = np.array(path_coords[0])
    source_coord = np.array(path_coords[-1])
    tip_to_source_vec = source_coord - tip_coord
    norm_tip_to_source = np.linalg.norm(tip_to_source_vec)
    if norm_tip_to_source == 0:
        return 0.0
    angles = []
    intersection_nodes = get_intersection_nodes(G, path_nodes)
    for node in intersection_nodes:
        node_coord = np.array(G.nodes[node]['coord'])
        tip_to_node_vec = node_coord - tip_coord
        norm_tip_to_node = np.linalg.norm(tip_to_node_vec)
        if norm_tip_to_node == 0:
            angle = 0.0
        else:
            cos_theta = np.clip(
                np.dot(tip_to_source_vec, tip_to_node_vec) /
                (norm_tip_to_source * norm_tip_to_node),
                -1.0, 1.0
            )
            angle = np.arccos(cos_theta)
        angles.append(angle)
    if angles:
        average_global_angle_deviation = sum(angle ** 2 for angle in angles) / len(angles)
    else:
        average_global_angle_deviation = 0.0
    return average_global_angle_deviation

def compute_local_angle_deviation(G, path_coords, path_nodes):
    intermed_indices = []
    for idx, node in enumerate(path_nodes):
        if G.nodes[node].get('label') == 'intermed':
            intermed_indices.append(idx)
    key_indices = [0] + intermed_indices + [len(path_nodes) - 1]
    key_indices = sorted(set(key_indices))
    angles = []
    for i in range(1, len(key_indices) - 1):
        idx_prev = key_indices[i - 1]
        idx_curr = key_indices[i]
        idx_next = key_indices[i + 1]
        coord_prev = np.array(path_coords[idx_prev])
        coord_curr = np.array(path_coords[idx_curr])
        coord_next = np.array(path_coords[idx_next])
        vec1 = coord_curr - coord_prev
        vec2 = coord_next - coord_curr
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            angle = 0.0
        else:
            cos_theta = np.clip(
                np.dot(vec1, vec2) / (norm1 * norm2),
                -1.0, 1.0
            )
            angle = np.arccos(cos_theta)
        angles.append(angle)
    if angles:
        average_local_angle_deviation = sum(angle ** 2 for angle in angles) / len(angles)
    else:
        average_local_angle_deviation = 0.0
    return average_local_angle_deviation

def compute_max_angle_deviation(path_coords):
    max_angle = 0.0
    for i in range(1, len(path_coords) - 1):
        coord_prev = np.array(path_coords[i - 1])
        coord_curr = np.array(path_coords[i])
        coord_next = np.array(path_coords[i + 1])
        vec1 = coord_curr - coord_prev
        vec2 = coord_next - coord_curr
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0.0 or norm2 == 0.0:
            angle = 0.0
        else:
            cos_theta = np.clip(
                np.dot(vec1, vec2) / (norm1 * norm2),
                -1.0, 1.0
            )
            angle = np.arccos(cos_theta)
        if angle > max_angle:
            max_angle = angle
    return max_angle

def compute_weighted_angle_deviation(G, path, path_coords, w_local=0.4, w_global=0.6):
    local_angle_dev = compute_local_angle_deviation(G, path_coords, path)
    global_angle_dev = compute_global_angle_deviation(G, path_coords, path)
    weighted_angle_dev = w_local * local_angle_dev + w_global * global_angle_dev
    max_angle_dev = compute_max_angle_deviation(path_coords)
    return weighted_angle_dev, max_angle_dev

def compute_max_angle_deviation_for_path(G, path):
    path_coords = [G.nodes[n]['coord'] for n in path]
    return compute_max_angle_deviation(path_coords)

def total_cost(G, selected_paths, final_paths_nodes, alpha=1.0, beta=2.0, gamma=1.0, w_local=0.4, w_global=0.6):
    total_iou = 0.0
    total_angle_dev = 0.0
    total_new_nodes = 0
    existing_nodes = final_paths_nodes.copy()
    for path in selected_paths:
        path_nodes = set(path)
        path_coords = [G.nodes[n]['coord'] for n in path]
        # Compute IoU with existing nodes
        iou = compute_iou(path_nodes, existing_nodes)
        total_iou += iou
        # Compute weighted angle deviation and max angle
        angle_dev, max_angle_dev = compute_weighted_angle_deviation(G, path, path_coords, w_local, w_global)
        total_angle_dev += angle_dev
        # Optionally, you can penalize paths that exceed the max angle
        # if max_angle_dev > MAX_ALLOWED_ANGLE_RAD:
        #     total_angle_dev += 1000  # High penalty
        # Count new nodes added by this path
        new_nodes = path_nodes - existing_nodes
        total_new_nodes += len(new_nodes)
        # Update existing nodes
        existing_nodes.update(path_nodes)
    # Compute total cost
    return alpha * total_iou + beta * total_angle_dev - gamma * total_new_nodes

def optimize_paths(G, final_paths, multiple_tips_paths, alpha=1.0, beta=2.0, gamma=1.0, w_local=0.4, w_global=0.6, initial_temp=1000, final_temp=1, cooling_rate=0.95, max_iter=1000):
    final_paths_nodes = set(node for path in final_paths for node in path)
    tips = list(multiple_tips_paths.keys())
    candidate_paths = [multiple_tips_paths[tip] for tip in tips]
    
    # # Filtra i percorsi candidati in base all'angolo massimo consentito
    # for i in range(len(candidate_paths)):
    #     paths_within_angle = []
    #     paths_exceeding_angle = []
    #     for path in candidate_paths[i]:
    #         max_angle = compute_max_angle_deviation_for_path(G, path)
    #         if max_angle <= MAX_ALLOWED_ANGLE_RAD:
    #             paths_within_angle.append(path)
    #         else:
    #             paths_exceeding_angle.append(path)
    #     if paths_within_angle:
    #         candidate_paths[i] = paths_within_angle
    #     else:
    #         # Se non ci sono percorsi entro l'angolo massimo, tieni quelli che lo superano
    #         candidate_paths[i] = paths_exceeding_angle  # Mantieni i percorsi che superano l'angolo massimo
    # # Dopo il filtraggio, assicuriamoci di avere almeno un percorso per ogni tip
    # for i in range(len(candidate_paths)):
    #     if not candidate_paths[i]:
    #         print(f"No paths available for tip {tips[i]}.")
    #         return []
    # Inizializza la soluzione corrente
    current_solution = [random.choice(paths) for paths in candidate_paths]
    best_solution = current_solution.copy()
    current_cost = total_cost(G, current_solution, final_paths_nodes, alpha, beta, gamma, w_local, w_global)
    best_cost = current_cost
    temperature = initial_temp
    iteration = 0
    while temperature > final_temp and iteration < max_iter:
        iteration += 1
        neighbor_solution = current_solution.copy()
        tip_index = random.randint(0, len(candidate_paths) - 1)
        tip_candidates = candidate_paths[tip_index]
        if len(tip_candidates) > 1:
            new_path = random.choice([path for path in tip_candidates if path != current_solution[tip_index]])
        else:
            new_path = tip_candidates[0]
        neighbor_solution[tip_index] = new_path
        neighbor_cost = total_cost(G, neighbor_solution, final_paths_nodes, alpha, beta, gamma, w_local, w_global)
        delta_cost = neighbor_cost - current_cost
        if delta_cost < 0 or random.uniform(0, 1) < math.exp(-delta_cost / temperature):
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
        temperature *= cooling_rate
    return best_solution

def select_best_paths(img, graph, valid_paths, final_paths, img_name):
    base_img = img.copy()
    for final_p in final_paths:
        color = (random.randint(0, 255), random.randint(0, 255))
        base_img = psu.print_single_path_on_image(base_img, final_p, graph, f'only1_path_root_{img_name}', color)
    paths_by_tip = defaultdict(list)
    for path in valid_paths:
        tip = path[0]
        paths_by_tip[tip].append(path)
    # Remove tips with no valid paths
    paths_by_tip = {tip: paths for tip, paths in paths_by_tip.items() if paths}
    if not paths_by_tip:
        print("No valid paths available.")
        return final_paths
    alpha = 1.0
    beta = 3.5
    gamma = 0.8
    w_local = 0.6
    w_global = 0.4
    best_paths = optimize_paths(graph, final_paths, paths_by_tip, alpha, beta, gamma, w_local, w_global)
    if not best_paths:
        print("No valid paths found after optimization.")
        return final_paths
    final_paths.extend(best_paths)
    for i, final_p in enumerate(best_paths):
        color = (random.randint(0, 255), random.randint(0, 255))
        img_name_without_ext = img_name.replace('.jpg', '')
        base_img = psu.print_single_path_on_image(base_img, final_p, graph, f'final_paths\\Final_path_root_{img_name_without_ext}_{i}', color)
    print("Final paths:", final_paths)
    return final_paths