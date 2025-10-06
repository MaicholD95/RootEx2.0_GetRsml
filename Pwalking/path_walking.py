# import math
# import networkx as nx
# from Path.root_path import TipPathsInfo
# import Pwalking.walking_utils as wu
# import copy 
# from collections import defaultdict


# def get_all_valids_paths(plant_img):
#     stashed_tip_paths_items = []
#     final_tip_paths = []
#     graph = plant_img.get_graph()
#     tips = plant_img.get_graph_tips()
#     sources= plant_img.get_graph_sources()

#     #create a dictionary to store the visited nodes foreach tip
#     for tip in tips:
#         tip_root_path = TipPathsInfo(tip)
#         end = False
#         #tip_root_path.add_current_path_node(tip)        
#         tip_root_path.add_walked_node_current_path(tip)
        
      
#         added = False
#         while(not end):
#             neighbors,source_found = wu.walk_to_neighbor(graph,tip_root_path)
#             #source found = end of the path
#             if len(source_found)>0:
#                 added = True
#                 tip_root_path.add_current_path_node(source_found[0])
#                 final_tip_paths.append(tip_root_path.get_current_path_nodes())
#                 #if there aren't any stashed paths, return
#                 if len(stashed_tip_paths_items) == 0 :
#                     end = True
#                     break
#                 #else get the stashed path and continue walking
#                 else:
#                     tip_root_path = get_first_stashed_path_info(stashed_tip_paths_items)
#             else:
#                 if len(neighbors) == 0:
#                     #if there aren't any stashed paths, return
#                     if len(stashed_tip_paths_items) == 0 :
#                         end = True
#                         break
#                     #if there are stashed paths, get the first one and continue walking replacing all the variables
#                     tip_root_path = get_first_stashed_path_info(stashed_tip_paths_items)
#                 elif len(neighbors) == 1:
#                     #if there is only one neighbor, walk to it
#                     set_next_node_info(tip_root_path, neighbors[0])

#                 elif len(neighbors) > 1:
#                     #if there are multiple neighbors, stash the current path and walk to the first neighbor
#                     stash_path_info(tip_root_path,neighbors[1],stashed_tip_paths_items)
#                     set_next_node_info(tip_root_path, neighbors[0])
        
#         #if the tip has not been added to the final paths, add the shortest path of short path between the tip and every source
#         if not added:
#             shortest_path = wu.get_shortest_path_from_tip_to_sources(graph,sources,tip)
#             final_tip_paths.append(shortest_path)
        
    
#     unique_tips_paths = []
#     multiple_tips_paths = []
#     #count the occurance of every tip in the final paths
#     start_count = defaultdict(list)
#     for sublist in final_tip_paths:
#         start_count[sublist[0]].append(sublist)
        
#     for key, lists in start_count.items():
#         if len(lists) == 1:
#             unique_tips_paths.extend(lists)
#         else:
#             multiple_tips_paths.extend(lists)

#     return multiple_tips_paths,unique_tips_paths
            
# #handle the new neighbor 
# def set_next_node_info(tip_root_path, neigh_node):
#     tip_root_path.add_current_path_node(neigh_node)
#     tip_root_path.add_walked_node_current_path(neigh_node)
#     current_node = neigh_node
#     return current_node                  
                    
# def stash_path_info(tip_root_path,next_neigh_node,stashed_tip_paths_items):
#     stashed_item = copy.deepcopy(tip_root_path)
#     stashed_item.add_current_path_node(next_neigh_node)
#     stashed_item.add_walked_node_current_path(next_neigh_node)
#     stashed_tip_paths_items.append(stashed_item)
    
# #load all the stashed path info on the current path
# def get_first_stashed_path_info(stashed_tip_paths_items):
#     stashed_item = stashed_tip_paths_items.pop(0)
#     return stashed_item   

    
import math
import networkx as nx
from Path.root_path import TipPathsInfo
import Pwalking.walking_utils as wu
import copy 
from collections import defaultdict


# def get_all_valids_paths(plant_img):
#     stashed_tip_paths_items = []
#     final_tip_paths = []
#     graph = plant_img.get_graph()
#     tips = plant_img.get_graph_tips()
#     sources= plant_img.get_graph_sources()

#     #create a dictionary to store the visited nodes foreach tip
#     for tip in tips:
#         tip_root_path = TipPathsInfo(tip)
#         end = False
#         #tip_root_path.add_current_path_node(tip)        
#         tip_root_path.add_walked_node_current_path(tip)
        
      
#         added = False
#         while(not end):
#             neighbors,source_found = wu.walk_to_neighbor(graph,tip_root_path)
#             #source found = end of the path
#             if len(source_found)>0:
#                 added = True
#                 tip_root_path.add_current_path_node(source_found[0])
#                 final_tip_paths.append(tip_root_path.get_current_path_nodes())
#                 #if there aren't any stashed paths, return
#                 if len(stashed_tip_paths_items) == 0 :
#                     end = True
#                     break
#                 #else get the stashed path and continue walking
#                 else:
#                     tip_root_path = get_first_stashed_path_info(stashed_tip_paths_items)
#             else:
#                 if len(neighbors) == 0:
#                     #if there aren't any stashed paths, return
#                     if len(stashed_tip_paths_items) == 0 :
#                         end = True
#                         break
#                     #if there are stashed paths, get the first one and continue walking replacing all the variables
#                     tip_root_path = get_first_stashed_path_info(stashed_tip_paths_items)
#                 elif len(neighbors) == 1:
#                     #if there is only one neighbor, walk to it
#                     set_next_node_info(tip_root_path, neighbors[0])

#                 elif len(neighbors) > 1:
#                     #if there are multiple neighbors, stash the current path and walk to the first neighbor
#                     stash_path_info(tip_root_path,neighbors[1],stashed_tip_paths_items)
#                     set_next_node_info(tip_root_path, neighbors[0])
        
#         #if the tip has not been added to the final paths, add the shortest path of short path between the tip and every source
#         if not added:
#             shortest_path = wu.get_shortest_path_from_tip_to_sources(graph,sources,tip)
#             final_tip_paths.append(shortest_path)
        
    
#     unique_tips_paths = []
#     multiple_tips_paths = []
#     #count the occurance of every tip in the final paths
#     start_count = defaultdict(list)
#     for sublist in final_tip_paths:
#         start_count[sublist[0]].append(sublist)
        
#     for key, lists in start_count.items():
#         if len(lists) == 1:
#             unique_tips_paths.extend(lists)
#         else:
#             multiple_tips_paths.extend(lists)

#     return multiple_tips_paths,unique_tips_paths
            
# #handle the new neighbor 
# def set_next_node_info(tip_root_path, neigh_node):
#     tip_root_path.add_current_path_node(neigh_node)
#     tip_root_path.add_walked_node_current_path(neigh_node)
#     current_node = neigh_node
#     return current_node                  
                    
# def stash_path_info(tip_root_path,next_neigh_node,stashed_tip_paths_items):
#     stashed_item = copy.deepcopy(tip_root_path)
#     stashed_item.add_current_path_node(next_neigh_node)
#     stashed_item.add_walked_node_current_path(next_neigh_node)
#     stashed_tip_paths_items.append(stashed_item)
    
# #load all the stashed path info on the current path
# def get_first_stashed_path_info(stashed_tip_paths_items):
#     stashed_item = stashed_tip_paths_items.pop(0)
#     return stashed_item   

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _neighbors_within_angle(G: nx.Graph, v, prev, theta_max):
    """
    Ritorna i vicini di `v` che rispettano il vincolo angolare θ_max
    rispetto all'arco (prev -> v).  
    Se `prev` è None (cioè v è il tip di partenza) nessun filtro viene applicato.
    """
    if prev is None:
        return list(G.neighbors(v))

    # direzione (prev -> v)
    x0, y0 = G.nodes[prev]['coord'][1], G.nodes[prev]['coord'][0]
    x1, y1 = G.nodes[v]['coord'][1], G.nodes[v]['coord'][0]
    vx, vy  = x1 - x0, y1 - y0           # vettore precedente

    valid = []
    for n in G.neighbors(v):
        if n == prev:                    # evita di tornare indietro
            continue
        x2, y2 = G.nodes[n]['coord'][1], G.nodes[n]['coord'][0]
        wx, wy  = x2 - x1, y2 - y1       # nuovo vettore
        # angolo tra (prev->v) e (v->n)
        dot   = vx * wx + vy * wy
        norm1 = math.hypot(vx, vy)
        norm2 = math.hypot(wx, wy)
        if norm1 == 0 or norm2 == 0:
            continue
        cos_angle = (vx * wx + vy * wy) / (norm1 * norm2)
        cos_angle = max(-1.0, min(1.0, cos_angle))   # <-- clamp
        angle = math.degrees(math.acos(cos_angle))
        if angle <= theta_max:
            valid.append(n)
    return valid


# ------------------------------------------------------------------------------
# DFS principale
# ------------------------------------------------------------------------------

def extract_valid_paths(plant_img, theta_max):
    """
    Implementazione one-shot della procedura `extractValidPaths`
    (vedi pseudocodice). Restituisce la tupla:
        (paths_multipli, paths_unici)
    dove ogni elemento è una lista di percorsi (liste di nodi).
    """
    G        = plant_img.get_graph()
    tips     = plant_img.get_graph_tips()
    sources  = set(plant_img.get_graph_sources())

    valid_paths   = []                  # insieme finale 𝓟_v
    paths_by_tip  = defaultdict(list)   # per distinguere unici / multipli

    # ------------------------------------------------------------------ DFS ---
    def dfs(v, path):
        """
        Ricorsione in profondità con back-tracking:
        - `v`  = nodo corrente
        - `path` = lista immutabile di nodi già percorsi
        Quando si raggiunge un source, il path viene salvato.
        """
        if v in sources:
            valid_paths.append(path)
            paths_by_tip[path[0]].append(path)
            return

        prev = path[-2] if len(path) > 1 else None
        for n in _neighbors_within_angle(G, v, prev, theta_max):
            if n not in path:           # evita cicli
                dfs(n, path + [n])

    # --------------------------------------------------------- main loop -----
    for tip in tips:
        dfs(tip, [tip])

        # Se non è stato salvato alcun path per questo tip
        if not paths_by_tip[tip]:
            shortest = wu.get_shortest_path_from_tip_to_sources(G, sources, tip)
            valid_paths.append(shortest)
            paths_by_tip[tip].append(shortest)

    # ---------------------------------------------------- split dei percorsi --
    unique_paths   = []
    multiple_paths = []
    for tip, plist in paths_by_tip.items():
        if len(plist) == 1:
            unique_paths.extend(plist)
        else:
            multiple_paths.extend(plist)

    return multiple_paths, unique_paths    