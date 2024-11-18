import cv2
from skimage.morphology import skeletonize
from skimage import io
import networkx as nx
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import networkx as nx
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import networkx as nx
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def build_graph_from_skeleton(skeleton):
    """
    Costruisce un grafo NetworkX da un'immagine skeletonizzata.
    Solo i punti con 1 o più di 2 vicini sono considerati nodi e vengono etichettati come 'external' o 'intersection'.
    
    Args:
        skeleton (np.ndarray): Immagine skeletonizzata (binary, valori 0 e 1).
    
    Returns:
        nx.MultiGraph: Grafo rappresentante lo skeleton con nodi etichettati.
    """
    G = nx.MultiGraph()  # Usa MultiGraph per permettere archi multipli tra nodi
    rows, cols= skeleton.shape

    # Definisci gli offset per i 8 vicini (8-connectività)
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0), (1, 1)]
    
    # Calcola il numero di vicini per ogni pixel usando una convoluzione
    structure = np.ones((3,3), dtype=int)
    neighbor_count = ndimage.convolve(skeleton.astype(int), structure, mode='constant', cval=0) - skeleton

    # Identifica i nodi
    external_nodes = set(zip(*np.where((skeleton == 1) & (neighbor_count == 1))))
    intersection_nodes = set(zip(*np.where((skeleton == 1) & (neighbor_count > 2))))
    all_nodes = external_nodes.union(intersection_nodes)

    # Aggiungi nodi al grafo con etichetta
    for node in external_nodes:
        G.add_node(node, label='external')
    for node in intersection_nodes:
        G.add_node(node, label='intersection')

    # Inizializza una matrice per tenere traccia dei pixel visitati
    visited_pixels = np.zeros_like(skeleton, dtype=bool)

    def traverse(node, direction):
        """
        Traccia un arco dallo skeleton in una direzione specifica fino a un altro nodo.
        
        Args:
            node (tuple): Coordinate del nodo di partenza (y, x).
            direction (tuple): Offset della direzione (dy, dx).
        
        Returns:
            list or None: Lista di coordinate che compongono l'arco, oppure None se non si trova un altro nodo.
        """
        path = [node]
        y, x = node
        dy, dx = direction
        current = (y + dy, x + dx)
        prev = node

        while True:
            cy, cx = current
            if not (0 <= cy < rows and 0 <= cx < cols):
                return None  # Fuori dai limiti
            if skeleton[cy, cx] == 0:
                return None  # Non è un pixel dello skeleton
            if visited_pixels[cy, cx]:
                return None  # Pixel già visitato
            
            path.append(current)
            visited_pixels[cy, cx] = True

            # Trova i vicini del pixel corrente escludendo il pixel precedente
            neighbors = []
            for offset in neighbor_offsets:
                ny, nx_ = cy + offset[0], cx + offset[1]
                neighbor = (ny, nx_)
                if (0 <= ny < rows and 0 <= nx_ < cols and
                    skeleton[ny, nx_] and neighbor != prev):
                    neighbors.append(neighbor)
            
            if len(neighbors) == 0:
                return None  # Fine dell'arco
            elif len(neighbors) > 1:
                # Se il pixel corrente è un nodo, termina l'arco
                if current in all_nodes:
                    return path
                else:
                    return None  # Ambiguità nel percorso
            else:
                next_node = neighbors[0]
                if next_node in all_nodes:
                    path.append(next_node)
                    return path
                else:
                    prev, current = current, next_node

    # Traccia gli archi da ogni nodo
    for node in all_nodes:
        y, x = node
        for offset in neighbor_offsets:
            dy, dx = offset
            neighbor = (y + dy, x + dx)
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                skeleton[neighbor] == 1 and neighbor not in all_nodes):
                if not visited_pixels[neighbor]:
                    path = traverse(node, offset)
                    if path and len(path) >= 2:
                        start_node = path[0]
                        end_node = path[-1]
                        # Calcola la lunghezza dell'arco
                        length = 0
                        for i in range(len(path)-1):
                            dy_ = path[i+1][0] - path[i][0]
                            dx_ = path[i+1][1] - path[i][1]
                            length += np.sqrt(dy_**2 + dx_**2)
                        # Aggiungi l'arco al grafo
                        G.add_edge(start_node, end_node, weight=length, path=path)

    print(f"Grafo costruito con {G.number_of_nodes()} nodi e {G.number_of_edges()} archi.")
    return G

def prune_external_nodes(graph, min_length=30):
    """
    Pruni i nodi 'external' e i relativi archi se la lunghezza degli archi è inferiore a min_length.
    Questa funzione viene applicata ricorsivamente fino a quando non rimangono nodi 'external' con archi corti.

    Args:
        graph (nx.MultiGraph): Il grafo da pulire.
        min_length (float): Lunghezza minima degli archi da conservare.

    Returns:
        nx.MultiGraph: Il grafo pulito senza nodi 'external' e i relativi archi.
    """
    pruned_graph = graph.copy()
    nodes_removed = True
    iteration = 0

    while nodes_removed:
        nodes_removed = False
        iteration += 1

        # Identifica i nodi 'external'
        external_nodes = [
            node for node, data in pruned_graph.nodes(data=True)
            if data.get('label') == 'external'
        ]

        # Identifica i nodi 'external' con archi di lunghezza < min_length
        nodes_to_remove = []
        for node in external_nodes:
            # I nodi 'external' dovrebbero avere solo un arco
            edges = list(pruned_graph.edges(node, keys=True, data=True))
            if len(edges) == 0:
                nodes_to_remove.append(node)  # Nodo isolato, rimuovi
                continue
            for edge in edges:
                length = edge[3].get('weight', 0)
                if length < min_length:
                    nodes_to_remove.append(node)
                    break  # Rimuovi il nodo se uno dei suoi archi è < min_length

        if nodes_to_remove:
            print(f"Removing {len(nodes_to_remove)} external nodes with edges shorter than {min_length}.")
            # Rimuovi i nodi identificati
            pruned_graph.remove_nodes_from(nodes_to_remove)
            nodes_removed = True
        else:
            print("No more external nodes to remove.")

    return pruned_graph

def merge_degree_two_intersections(graph):
    """
    Trasforma i nodi 'intersection' con esattamente 2 vicini in archi diretti tra i loro vicini,
    seguendo il percorso dello skeleton.

    Args:
        graph (nx.MultiGraph): Il grafo da modificare.

    Returns:
        nx.MultiGraph: Il grafo modificato senza i nodi 'intersection' con 2 vicini.
    """
    merged = True
    while merged:
        merged = False
        # Identifica i nodi 'intersection' con grado esattamente 2
        nodes_to_merge = [
            node for node, data in graph.nodes(data=True)
            if data.get('label') == 'intersection' and graph.degree(node) == 2
        ]
        
        if not nodes_to_merge:
            break  # Nessun nodo da fondere
        
        for node in nodes_to_merge:
            neighbors = list(graph.neighbors(node))
            if len(neighbors) != 2:
                continue  # Salta se non ha esattamente 2 vicini
            
            u, v = neighbors
            # Ottenere tutti gli archi tra node e u e tra node e v
            edges_u = list(graph.get_edge_data(node, u, default={}).items())  # Lista di (key, data_dict)
            edges_v = list(graph.get_edge_data(node, v, default={}).items())
            
            # Assicurati che ci sia almeno un arco tra node e u e node e v
            if not edges_u or not edges_v:
                continue
            
            # Prendi il primo arco disponibile tra node e u e tra node e v
            key_u, data_u = edges_u[0]
            key_v, data_v = edges_v[0]
            
            path_u = data_u.get('path', [])
            path_v = data_v.get('path', [])
            
            # Evita di unire archi senza percorsi validi
            if not path_u or not path_v:
                continue
            
            # Verifica la direzione dei percorsi
            if path_u[0] == u and path_u[-1] == node:
                ordered_path_u = path_u
            elif path_u[0] == node and path_u[-1] == u:
                ordered_path_u = path_u[::-1]
            else:
                continue  # Percorso non valido
            
            if path_v[0] == node and path_v[-1] == v:
                ordered_path_v = path_v
            elif path_v[0] == v and path_v[-1] == node:
                ordered_path_v = path_v[::-1]
            else:
                continue  # Percorso non valido
            
            # Costruisci il nuovo percorso includendo il nodo intermedio
            new_path = ordered_path_u[:-1] + [node] + ordered_path_v[1:]
            
            # Calcola la nuova lunghezza
            length = 0
            for i in range(len(new_path)-1):
                dy = new_path[i+1][0] - new_path[i][0]
                dx = new_path[i+1][1] - new_path[i][1]
                length += np.sqrt(dy**2 + dx**2)
            
           
            # Aggiungi il nuovo arco tra u e v
            graph.add_edge(u, v, weight=length, path=new_path)
            
            # Rimuovi gli archi originali utilizzando le chiavi corrette
            graph.remove_edge(node, u, key=key_u)
            graph.remove_edge(node, v, key=key_v)
            
            # Rimuovi il nodo 'intersection' fuso
            graph.remove_node(node)
            
            merged = True
            break  # Esci dai loop per ricominciare
        
    print("Completed merging of degree-two intersection nodes.")
    return graph

def visualize_graph(skeleton, graph, plant_name, save_path=None, show_node_types=False):
    """
    Visualizza gli archi del grafo con spessore di 1 pixel. Se show_node_types è True,
    visualizza anche i nodi con colori differenti in base al loro tipo.

    Args:
        skeleton (np.ndarray): Immagine skeletonizzata (binary).
        graph (nx.MultiGraph): Grafo da visualizzare.
        plant_name (str): Nome della pianta per il titolo del grafico.
        save_path (str, optional): Percorso per salvare l'immagine del grafo. Se None, mostra il grafico.
        show_node_types (bool, optional): Se True, visualizza i nodi con colori differenti.
    """
    # Crea un'immagine vuota RGB con le stesse dimensioni dello skeleton
    graph_image = np.zeros((*skeleton.shape, 3), dtype=np.uint8)  # Immagine RGB

    # Disegna gli archi in bianco
    for u, v, key, data in graph.edges(keys=True, data=True):
        path = data.get('path', [])
        for y, x in path:
            if 0 <= y < graph_image.shape[0] and 0 <= x < graph_image.shape[1]:
                graph_image[y, x] = (255, 255, 255)  # Bianco

    if show_node_types:
        # Disegna i nodi con colori differenti in base al loro tipo
        for node, data in graph.nodes(data=True):
            y, x = node
            if 0 <= y < graph_image.shape[0] and 0 <= x < graph_image.shape[1]:
                label = data.get('label')
                if label == 'external':
                    graph_image[y, x] = (255, 0, 0)  # Rosso
                elif label == 'intersection':
                    graph_image[y, x] = (0, 255, 0)  # Verde
                else:
                    graph_image[y, x] = (0, 0, 255)  # Blu per altri tipi di nodi
    else:
        # Disegna i nodi in bianco (opzionale)
        for node in graph.nodes():
            y, x = node
            if 0 <= y < graph_image.shape[0] and 0 <= x < graph_image.shape[1]:
                graph_image[y, x] = (255, 255, 255)
        #return the image in grayscale
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_BGR2GRAY)


    if save_path:
        io.imsave(save_path, graph_image)
        print(f"Graph visualization saved to {save_path}")
    
    return graph_image

def get_pruned_skeleton_graph(img_name = "",skeleton = None,saving_path = None):
    img_name = img_name.replace('.jpg','')
    #flipped_skel = np.flipud(skeleton)
    flipped_skel = skeleton
    image = flipped_skel > 0.5
    skeleton = skeletonize(image > 0).astype(np.uint8)
    graph = build_graph_from_skeleton(skeleton)
    
    # Prune dei nodi 'external' con una soglia minima di 30
    pruned_graph = prune_external_nodes(graph, min_length=30)
    # Fonde i nodi 'intersection' con grado 2
    merged_graph = merge_degree_two_intersections(pruned_graph)
    # Visualizza il grafo pruned e fuso
    pruned_skeleton_img = visualize_graph(skeleton, merged_graph, plant_name=img_name)
    pruned_skeleton_img = skeletonize(pruned_skeleton_img > 0).astype(np.uint8)

    graph = build_graph_from_skeleton(pruned_skeleton_img)
    visualize_graph(pruned_skeleton_img, graph, plant_name=img_name, save_path=f'{saving_path}\\skeleton_{img_name}_pp.png',show_node_types=True)
    
    return graph
# if __name__ == "__main__":
#     # Percorso all'immagine skeletonizzata
#     image_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\skeleton_130R.jpg.png'
    
#     # Carica l'immagine
#     image = io.imread(image_path, as_gray=True)
#     #binarize the image
#     image = image > 0.5
#     skeleton = skeletonize(image > 0).astype(np.uint8)
    
#     # Costruisci il grafo
#     graph = build_graph_from_skeleton(skeleton)
    
#     # Pruni i nodi 'external' con una soglia minima di 2
#     pruned_graph = prune_external_nodes(graph, min_length=30)
    
#     # Fonde i nodi 'intersection' con grado 2
#     merged_graph = merge_degree_two_intersections(pruned_graph)
    
#     # Visualizza il grafo pruned e fuso
#     pruned_skeleton_img = visualize_graph(skeleton, merged_graph, plant_name='TestPlant', save_path='test_graph.png')
    
#     pruned_skeleton_img = skeletonize(pruned_skeleton_img > 0).astype(np.uint8)

#     graph = build_graph_from_skeleton(pruned_skeleton_img)
#     visualize_graph(pruned_skeleton_img, graph, plant_name='TestPlant', save_path='test_graph_p.png',show_node_types=True)
