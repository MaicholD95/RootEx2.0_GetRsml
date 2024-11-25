import Path_selection.path_selection_utils as psu
import random
import itertools
import numpy as np
from collections import defaultdict
import math

def compute_iou(path_nodes, existing_nodes):
    intersection = path_nodes & existing_nodes
    union = path_nodes | existing_nodes
    return len(intersection) / len(union) if union else 0.0

def compute_cumulative_angle_deviation(path_coords):
    angles = []
    for i in range(1, len(path_coords) - 1):
        vec1 = np.array(path_coords[i]) - np.array(path_coords[i - 1])
        vec2 = np.array(path_coords[i + 1]) - np.array(path_coords[i])
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            angle = 0
        else:
            cos_theta = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cos_theta)
        angles.append(angle)
    total_angle_deviation = sum(abs(angle) for angle in angles)
    return total_angle_deviation

def total_cost(G, paths, covered_nodes, alpha=1.0, beta=5.0, gamma=1.0):
    total_iou = 0.0
    total_angle_dev = 0.0
    total_new_nodes = 0.0
    existing_nodes = covered_nodes.copy()
    for path in paths:
        path_nodes = set(path)
        path_coords = [G.nodes[n]['coord'] for n in path]
        # Compute IoU with existing nodes
        iou = compute_iou(path_nodes, existing_nodes)
        total_iou += iou
        # Compute angle deviation
        angle_dev = compute_cumulative_angle_deviation(path_coords)
        total_angle_dev += angle_dev
        # Compute new nodes added by this path
        new_nodes = path_nodes - existing_nodes
        total_new_nodes += len(new_nodes)
        # Update existing nodes
        existing_nodes.update(path_nodes)
    # Calculate total cost
    cost = alpha * total_iou + beta * total_angle_dev - gamma * total_new_nodes
    return cost

def simulated_annealing(G, final_paths, multiple_tips_paths, alpha=1.0, beta=5.0, gamma=1.0, initial_temp=1000, final_temp=1, cooling_rate=0.95, max_iter=1000):
    # Initialize covered nodes with nodes from final_paths
    covered_nodes = set(node for path in final_paths for node in path)
    # Group paths by tip
    paths_by_tip = defaultdict(list)
    for path in multiple_tips_paths:
        tip = path[0]
        paths_by_tip[tip].append(path)
    tips = list(paths_by_tip.keys())
    # Initialize with random paths for each tip
    current_solution = []
    for tip in tips:
        current_solution.append(random.choice(paths_by_tip[tip]))
    best_solution = current_solution.copy()
    # Calculate initial cost
    combined_paths = final_paths + current_solution
    current_cost = total_cost(G, combined_paths, covered_nodes, alpha, beta, gamma)
    best_cost = current_cost
    temperature = initial_temp
    iteration = 0
    while temperature > final_temp and iteration < max_iter:
        iteration += 1
        # Generate neighbor solution
        neighbor_solution = current_solution.copy()
        # Randomly select a tip to modify its path
        tip_index = random.randint(0, len(tips) - 1)
        tip = tips[tip_index]
        # Exclude current path from candidates
        candidate_paths = [p for p in paths_by_tip[tip] if p != current_solution[tip_index]]
        if candidate_paths:
            new_path = random.choice(candidate_paths)
            neighbor_solution[tip_index] = new_path
            # Calculate cost of neighbor solution
            combined_paths = final_paths + neighbor_solution
            neighbor_cost = total_cost(G, combined_paths, covered_nodes, alpha, beta, gamma)
            # Decide whether to accept the neighbor solution
            delta_cost = neighbor_cost - current_cost
            if delta_cost < 0 or random.uniform(0, 1) < math.exp(-delta_cost / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
        # Cooling schedule
        temperature *= cooling_rate
    # Combine best solution with final_paths
    final_paths.extend(best_solution)
    return final_paths

def filter_paths_by_angle_deviation(G, multiple_tips_paths, max_angle_deviation):
    filtered_paths = []
    for path in multiple_tips_paths:
        path_coords = [G.nodes[n]['coord'] for n in path]
        angle_dev = compute_cumulative_angle_deviation(path_coords)
        if angle_dev <= max_angle_deviation:
            filtered_paths.append(path)
    return filtered_paths


def select_best_paths(img,graph,valid_paths,final_paths,img_name):
    base_img = img
    for final_p in final_paths:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        base_img = psu.print_single_path_on_image(base_img,final_p,graph,f'only1_path_root_{img_name}',color)
    
    alpha = 5.0    # Weight for overlap penalty
    beta = 7.0     # Weight for angle deviation penalty
    gamma = 7.0   # Weight for coverage reward   
    
    final_paths = simulated_annealing(graph, final_paths, valid_paths, alpha, beta, gamma)
    new_paths = [path for path in valid_paths if path  in final_paths]
    for final_p in new_paths:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        base_img = psu.print_single_path_on_image(base_img,final_p,graph,f'Final_path_root_{img_name}',color)
        
    print("final_paths",final_paths)