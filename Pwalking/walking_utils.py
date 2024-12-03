import math
import networkx as nx




def get_shortest_path_from_tip_to_sources(graph,sources,tip):
    paths = []
    for source in sources:
        if nx.has_path(graph,tip,source):
            path = nx.shortest_path(graph,tip,source)
            paths.append(path)
    #get the shortest path
    shortest_path = min(paths,key=len)
    return shortest_path

#walk to a neighbor node
def walk_to_neighbor(graph,tip_root_path):
    current_node = tip_root_path.get_current_path_last_node()
    walked_nodes = tip_root_path.get_current_path_walked_nodes()
    
    neighbors = list(graph.neighbors(current_node))
    #remove the nodes that have been visited
    neighbors = [node for node in neighbors if node not in walked_nodes]
    #if the current point y is over the neighbor node + threshold, remove the neighbor node
    threshold = 10
    #get current node coords
    current_x, current_y = graph.nodes[current_node]['coord'][1], graph.nodes[current_node]['coord'][0]
    neighbors = [node for node in neighbors if graph.nodes[node]['coord'][0] < current_y + threshold]

    #check if some sources are in the neighbors
    sources = [node for node in neighbors if graph.nodes[node]['label'] == 'source']
        
    return neighbors,sources

def calculate_angle(p1, p2):
    """
    Calculate the angle between two points in an image where to the left of the point the angle is < 0,-90 and to the right is >0,90.

    Args:
        p1 (tuple): The first point (x, y).
        p2 (tuple): The second point (x, y).

    Returns:
        float: The angle between the two points in degrees.
    """
    delta_x = p2[1] - p1[1]
    delta_y = p2[0] - p1[0]
    
    angle = math.degrees(math.atan2(delta_y, delta_x))
    
    # # Adjust the angle based on the relative positions of the points
    # if delta_x < 0:
    #     angle -= 180
    # elif delta_x == 0 and delta_y < 0:
    #     angle = -90
    # elif delta_x == 0 and delta_y > 0:
    #     angle = 90

    return angle

