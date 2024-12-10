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
#from Graph.graph import get_pruned_skeleton_graph,divide_paths_with_equidistant_nodes,visualize_graph
from Skeleton.skeleton import get_skeleton
from Skeleton.sknw import build_sknw
from skimage.morphology import skeletonize
from skimage import io
import networkx as nx
from Pwalking.path_walking import get_all_valids_paths
#from Graph.skeleton_utils import get_skeleton
from Graph.graph_processing import get_pruned_skeleton_graph,get_nearest_node_in_range
from Graph.graph_utils import divide_paths_with_equidistant_nodes
from Graph.visualization import visualize_graph,print_graph_on_original_img
import Path_selection.path_selection as ps  
import json
from scipy.spatial import cKDTree
def visualize_associated_paths_on_image(original_image, associated_paths, graph, saving_path):
    """
    Visualizza i percorsi associati sul'immagine originale.

    Args:
        original_image (numpy.ndarray): L'immagine originale.
        associated_paths (list): Lista di percorsi associati, ogni percorso è una lista di ID dei nodi del grafo.
        graph (networkx.Graph): Il grafo.
        saving_path (str): Percorso per salvare l'immagine visualizzata.
    """
    for path in associated_paths:
        if not path:
            continue  # Salta i percorsi vuoti
        # Assegna un colore casuale a ogni percorso
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # Estrai le coordinate dei nodi nel percorso
        path_coords = [graph.nodes[node]['coord'] for node in path]
        # Disegna linee tra nodi consecutivi
        for i in range(len(path_coords) - 1):
            pt1 = (int(path_coords[i][1]), int(path_coords[i][0]))  # (x, y)
            pt2 = (int(path_coords[i+1][1]), int(path_coords[i+1][0]))
            cv2.line(original_image, pt1, pt2, color, 2)
    # Salva l'immagine annotata
    cv2.imwrite(saving_path, original_image)
    
def associate_gt_points_to_graph_nodes(graph, root_points, sources, tips, distance_threshold=10):
    """
    Associa ogni punto di ground truth al nodo più vicino nel grafo, assicurando che ogni percorso inizi con una source
    e termini con una tip. Se il primo nodo associato non è una source, trova il percorso più breve dalla source
    più vicina e preprende tale segmento al percorso.

    Args:
        graph (networkx.Graph): Il grafo contenente i nodi con attributo 'coord'.
        root_points (list): Lista di percorsi di ground truth, ogni percorso è una lista di tuple (x, y).
        sources (list): Lista di nodi del grafo etichettati come 'source'.
        tips (list): Lista di nodi del grafo etichettati come 'tip'.
        distance_threshold (float): La distanza massima per considerare un nodo come vicino.

    Returns:
        list: Lista di percorsi, ogni percorso è una lista di ID dei nodi del grafo che iniziano con una source e finiscono con una tip.
    """
    # Estrai le coordinate dei nodi dal grafo
    graph_node_coords = [graph.nodes[node]['coord'] for node in graph.nodes()]
    graph_node_ids = list(graph.nodes())

    # Costruisci un KD-Tree per i nodi del grafo
    tree = cKDTree(graph_node_coords)

    # Prepara un set di ID delle sources e tips per un accesso rapido
    source_ids = set(sources)
    tip_ids = set(tips)

    # Lista per memorizzare i percorsi associati
    associated_paths = []

    for path_idx, path in enumerate(root_points):
        associated_path = []
        for point in path:
            x, y = point
            distance, node_idx = tree.query([y, x], distance_upper_bound=distance_threshold)
            if distance != float('inf'):
                node_id = graph_node_ids[node_idx]
                if node_id not in associated_path:
                    associated_path.append(node_id)
            else:
                print(f"Per il punto ({x}, {y}) nel percorso {path_idx + 1}, nessun nodo trovato entro {distance_threshold} pixel.")

        # Verifica se il primo nodo è una source
        if associated_path and associated_path[0] not in source_ids:
            first_node = associated_path[0]
            # Trova la source più vicina al primo nodo
            nearest_source = None
            min_length = float('inf')
            for source in sources:
                try:
                    path_to_source = nx.shortest_path(graph, source=source, target=first_node)
                    if len(path_to_source) < min_length:
                        min_length = len(path_to_source)
                        nearest_source = source
                except nx.NetworkXNoPath:
                    continue

            if nearest_source:
                # Prependere il percorso dalla source al primo nodo
                path_segment = nx.shortest_path(graph, source=nearest_source, target=first_node)
                # Evita duplicati: se il primo nodo del percorso segmentale è già il primo nodo associato, salta
                if path_segment[0] != associated_path[0]:
                    associated_path = path_segment + associated_path
                else:
                    associated_path = path_segment
                print(f"Percorso {path_idx + 1}: Prependere il percorso dalla source {nearest_source} al nodo {first_node}.")
            else:
                print(f"Percorso {path_idx + 1}: Nessuna source trovata per prependere al nodo {first_node}.")

        associated_paths.append(associated_path)
    
    return associated_paths

def calculate_min_distance(path_coords, root_points_tree):
    """
    Calculate the minimum distance from the path to any root point using a KD-Tree for efficiency.
    """
    if not path_coords:
        return float('inf')
    # Convert path coordinates to a NumPy array
    path_array = np.array(path_coords)
    # Query the KD-Tree for the nearest root point to each point in the path
    distances, _ = root_points_tree.query(path_array, k=1)
    # Return the minimum distance found
    return distances.min()


def move_points_to_nearest_node(graph,sources,tips,s_distance_threshold=5, t_distance_threshold=5):

    def move_sources():
        for i,source in enumerate(sources):
            nearest_node_source_id = get_nearest_node_in_range(graph, source,radius_min=0, radius_max=s_distance_threshold)
            if nearest_node_source_id != -1:
                new_sources = sources
                new_sources[i] = graph.nodes[nearest_node_source_id]['coord']
                graph.nodes[nearest_node_source_id]['label'] = 'source'
            else:
                print(f"Source {source} not found in the graph")
            
    def move_tips():
        for i,tip in enumerate(tips):
            nearest_node_tip_id = get_nearest_node_in_range(graph, tip,radius_min=0, radius_max=t_distance_threshold)
            if nearest_node_tip_id != -1:
                new_tips = tips
                new_tips[i] = graph.nodes[nearest_node_tip_id]['coord']
                graph.nodes[nearest_node_tip_id]['label'] = 'tip'
            else:
                print(f"Tip {tip} not found in the graph")


    move_sources()
    move_tips()
    return graph

def move_general_points_to_nearest_node(graph,points,distance_threshold=5):
    for i,point in enumerate(points):
        nearest_node_id = get_nearest_node_in_range(graph, point,radius_min=0, radius_max=distance_threshold)
        if nearest_node_id != -1:
            points[i] = graph.nodes[nearest_node_id]['coord']
        else:
            print(f"Point {point} not found in the graph")
    return points

def print_graph_on_original_img(name,image_path,graph,sources,tips, saving_path):
    #overlap the graph (with nodes ecc) on the original image
    #get the original image
    original_image = io.imread(image_path)
    #get the nodes
    nodes = list(graph.nodes())
    #get the edges
    # Disegna gli edge utilizzando i dati del 'path'
    for u, v, data in graph.edges(data=True):
        path = data.get('path', [])
        if path:
            y_coords = [coord[0] for coord in path]
            x_coords = [coord[1] for coord in path]
            for x, y in zip(x_coords, y_coords):
                cv2.circle(original_image, (int(x),int(y)), 1, (0, 255, 255), -1)
                
    #draw the nodes
    for node in nodes:
        if node in sources:
            color = (0,255,0)
        elif node in tips:
            color = (255,0,0)
        else:
            color = (0,0,255)
        cv2.circle(original_image,(int(graph.nodes[node]['coord'][1]),int(graph.nodes[node]['coord'][0])),3,color,-1)
    


    #save the image
    cv2.imwrite(f'{saving_path}\\overlapped_{name}.png', original_image)




if __name__ == "__main__":
    gt_masks_path = r'C:\Users\maich\Desktop\rootex3\gt_masks'
    skeletons_saving_path = r'C:\Users\maich\Desktop\rootex3\gt_skeleton'
    gt_json_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\test_images'
    plant_imgs = []
    #read all gt masks
    gt_masks = glob(os.path.join(gt_masks_path, '*.png'))
    #read all json files
    gt_json = glob(os.path.join(gt_json_path, '*.json'))
    imgs = glob(os.path.join(gt_json_path, '*.jpg'))
    for index in range(len(gt_masks)):
        print(f"\nProcessing image {index+1}/{len(gt_masks)}")
        
        gt_og_image = cv2.imread(imgs[index])
        # Predict and visualize the segmentation for the current image in the test dataset
        gt_mask_path = gt_masks[index]
        gt_image = cv2.imread(gt_mask_path,cv2.IMREAD_GRAYSCALE)
        #get all the tips and sources from the json file
        gt_source = []
        gt_tip = []
        root_points = []
        for json_file in gt_json:
            if json_file.split('\\')[-1].split('.')[0] == gt_mask_path.split('\\')[-1].split('.')[0]:
                with open(json_file) as f:
                    data = json.load(f)
                    for image_info in data:
                        gt_tip.append((int(image_info['tip']['x']), int(image_info['tip']['y'])))
                        source = (int(image_info['source']['x']), int(image_info['source']['y']))
                        if source not in gt_source:
                            gt_source.append(source)
                        
                        all_x = image_info['root']['points']['all_points_x']
                        all_y = image_info['root']['points']['all_points_y']
                        path = list(zip(all_x, all_y))  # Crea una lista di tuple (x, y) per il percorso
                        root_points.append(path)
        ## skeletonization ##
        # Obtain the skeleton from the predicted mask and convert it to an 8-bit unsigned integer image
        skeleton = get_skeleton(gt_image).astype(np.uint8) * 255

        # Save the skeleton image with a filename based on the plant image name
        cv2.imwrite(f'gt_skeleton_{gt_image}.png', skeleton)
        img_name = gt_mask_path.split('\\')[-1].split('.')[0]
        ### Graph creation and processing ####
        graph = get_pruned_skeleton_graph(
            img_name,
            skeleton,
            saving_path=f'{skeletons_saving_path}\\gt_pruned_{img_name}.png'
        )
        
        #remove empty nodes from the graph
        graph.remove_nodes_from(list(nx.isolates(graph)))
        # Divide paths in the graph by adding equidistant intermediate nodes every 30 pixels
        graph = divide_paths_with_equidistant_nodes(graph,80)
        visualize_graph(
            skeleton,
            graph,
            img_name,
            save_path=f'{skeletons_saving_path}\\intermed_{img_name}.png',
            show_node_types=True
        )
        #remove empty nodes from the graph
        graph.remove_nodes_from(list(nx.isolates(graph)))
        # Move the sources and tips to the nearest node in the graph within a distance threshold of 5 pixels
        graph = move_points_to_nearest_node(graph,gt_source,gt_tip,120,100)
        visualize_graph(
            skeleton,
            graph,
            img_name,
            save_path=f'{skeletons_saving_path}\\moved_{img_name}.png',
            show_node_types=True
        )
        
        #remove empty nodes from the graph
        graph.remove_nodes_from(list(nx.isolates(graph)))        
        sources = [node for node in graph.nodes() if graph.nodes[node]['label'] == 'source']
        tips = [node for node in graph.nodes() if graph.nodes[node]['label'] == 'tip']
        for node in list(graph.nodes()):
            found = False
            for source in sources:
                if nx.has_path(graph,source,node):
                    found = True
                    break
            if not found:
                graph.remove_node(node)
        # Visualize the updated graph overlaid on the skeleton image
        visualize_graph(
            skeleton,
            graph,
            img_name,
            save_path=f'{skeletons_saving_path}\\removed_isolated_{img_name}.png',
            show_node_types=True
        )
        #remove empty nodes from the graph
        graph.remove_nodes_from(list(nx.isolates(graph)))
        # **Associazione dei punti di ground truth ai nodi del grafo**
        associated_paths = associate_gt_points_to_graph_nodes(graph, root_points,sources,tips,distance_threshold=8)
        print(f"Percorsi associati per l'immagine {img_name}: {len(associated_paths)}")
       
        visualize_associated_paths_on_image(
            gt_og_image,
            associated_paths,
            graph,
            saving_path=f'{skeletons_saving_path}\\associated_paths_{img_name}.png'
        )            
        
        json_saving_path = os.path.join(skeletons_saving_path, f'associated_paths_{img_name}.json')
        association_output = {
            'image_name': img_name,
            'associated_paths': []
        }

        for path in associated_paths:
            if path:  # Ignora percorsi vuoti
                # Ottieni le coordinate del tip per il percorso
                #invert the coordinates
                tip_coord = tuple(map(int, graph.nodes[path[-1]]['coord'][::-1]))  # Converte in tuple di int
                #tip_coord = tuple(map(int, graph.nodes[path[-1]]['coord']))  # Converte in tuple di int
                # Ottieni tutte le coordinate del percorso
                path_coords = [tuple(map(int, graph.nodes[node]['coord'])) for node in path]
                # Aggiungi il dizionario per questo percorso
                association_output['associated_paths'].append({
                    'tip_coord': tip_coord,
                    'path_coords': path_coords
                })

        with open(json_saving_path, 'w') as json_file:
            json.dump(association_output, json_file, indent=4)

        print(f"Percorsi associati salvati in {json_saving_path}")

        
        
        
        
        # ### path walking ###
        # final_paths = []
        # multiple_tips_paths = []
        # final_paths = []
        # #First i want all the valid paths
        # multiple_tips_paths,final_paths = get_all_valids_paths(plant_img)
        # #need to decide which path to keep
        # img = cv2.imread(plant_img.get_image_path())
        # ps.select_best_paths(img,plant_img.get_graph(),multiple_tips_paths,final_paths,plant_img.get_name())
        
         


