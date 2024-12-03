import math
import networkx as nx
from Path.root_path import TipPathsInfo
import Pwalking.walking_utils as wu
import copy 
from collections import defaultdict


def get_all_valids_paths(plant_img):
    stashed_tip_paths_items = []
    final_tip_paths = []
    graph = plant_img.get_graph()
    tips = plant_img.get_graph_tips()
    sources= plant_img.get_graph_sources()

    #create a dictionary to store the visited nodes foreach tip
    for tip in tips:
        if tip >=50 and tip <=60:
            print("tip 52 or 53")
        tip_root_path = TipPathsInfo(tip)
        end = False
        #tip_root_path.add_current_path_node(tip)        
        tip_root_path.add_walked_node_current_path(tip)
        
      
        added = False
        while(not end):
            neighbors,source_found = wu.walk_to_neighbor(graph,tip_root_path)
            #source found = end of the path
            if len(source_found)>0:
                print("------ source found ------")
                added = True
                tip_root_path.add_current_path_node(source_found[0])
                final_tip_paths.append(tip_root_path.get_current_path_nodes())
                #if there aren't any stashed paths, return
                if len(stashed_tip_paths_items) == 0 :
                    end = True
                    break
                #else get the stashed path and continue walking
                else:
                    tip_root_path = get_first_stashed_path_info(stashed_tip_paths_items)
            else:
                if len(neighbors) == 0:
                    print("no neighbors")
                    #if there aren't any stashed paths, return
                    if len(stashed_tip_paths_items) == 0 :
                        end = True
                        break
                    #if there are stashed paths, get the first one and continue walking replacing all the variables
                    tip_root_path = get_first_stashed_path_info(stashed_tip_paths_items)
                elif len(neighbors) == 1:
                    #if there is only one neighbor, walk to it
                    set_next_node_info(tip_root_path, neighbors[0])

                elif len(neighbors) > 1:
                    #if there are multiple neighbors, stash the current path and walk to the first neighbor
                    stash_path_info(tip_root_path,neighbors[1],stashed_tip_paths_items)
                    set_next_node_info(tip_root_path, neighbors[0])
        
        #if the tip has not been added to the final paths, add the shortest path of short path between the tip and every source
        if not added:
            print("tip_root_path not added")
            shortest_path = wu.get_shortest_path_from_tip_to_sources(graph,sources,tip)
            final_tip_paths.append(shortest_path)
            print("tip_root_path added")            
        
        print("tip_root_path.completed_paths")  
    
    print("got all valid paths")  
    unique_tips_paths = []
    multiple_tips_paths = []
    #count the occurance of every tip in the final paths
    start_count = defaultdict(list)
    for sublist in final_tip_paths:
        start_count[sublist[0]].append(sublist)
        
    for key, lists in start_count.items():
        if len(lists) == 1:
            unique_tips_paths.extend(lists)
        else:
            multiple_tips_paths.extend(lists)

    return multiple_tips_paths,unique_tips_paths
            
#handle the new neighbor 
def set_next_node_info(tip_root_path, neigh_node):
    tip_root_path.add_current_path_node(neigh_node)
    tip_root_path.add_walked_node_current_path(neigh_node)
    current_node = neigh_node
    return current_node                  
                    
def stash_path_info(tip_root_path,next_neigh_node,stashed_tip_paths_items):
    stashed_item = copy.deepcopy(tip_root_path)
    stashed_item.add_current_path_node(next_neigh_node)
    stashed_item.add_walked_node_current_path(next_neigh_node)
    stashed_tip_paths_items.append(stashed_item)
    
#load all the stashed path info on the current path
def get_first_stashed_path_info(stashed_tip_paths_items):
    stashed_item = stashed_tip_paths_items.pop(0)
    return stashed_item   

    