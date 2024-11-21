

#create a root path info class

class TipPathsInfo:

    def __init__(self, tip):
        self.tip = tip
        self.current_path_nodes = [tip]
        #self.stash_item= []
        self.best_path = None
        self.completed_paths = []
        self.current_path_walked_nodes = []
    
    def get_best_path(self):
        return self.best_path    

    def get_tip(self):
        return self.tip
    def get_current_path_nodes(self):
        return self.current_path_nodes
    def get_current_path_node_at_index(self,index):
        return self.current_path_nodes[index]
    def get_current_path_last_node(self):
        return self.current_path_nodes[-1]
    def get_current_path_first_node(self):
        return self.current_path_nodes[0]
    def get_current_path_node_count(self):
        return len(self.current_path_nodes)
    def get_current_path_walked_nodes(self):
        return self.current_path_walked_nodes
    
    def set_current_path_walked_nodes(self,walked_nodes):
        self.current_path_walked_nodes = walked_nodes
        
    def set_best_path(self,best_path):
        self.best_path = best_path  

    def set_tip(self,tip):
        self.tip = tip
    def add_current_path_node(self,node):
        self.current_path_nodes.append(node)
    def set_current_path_nodes(self,nodes):
        self.current_path_nodes = nodes

    # def add_stash_item(self,item):
    #     self.stash_item.append(item)
        
    # def pop_first_stash_path(self):
    #     return self.stash_item.pop(0)
    # def get_stashed_items_count(self):
    #     return len(self.stash_item)
    def add_walked_node_current_path(self,node):
        self.current_path_walked_nodes.append(node)
    
    
    def __str__(self):
        return f"Path: {self.path}, Source: {self.source}, Tip: {self.tip}, Length: {self.length}"
    
    def __repr__(self):
        return f"Path: {self.path}, Source: {self.source}, Tip: {self.tip}, Length: {self.length}"
    
    def clone(self):
        new_path = TipPathsInfo(self.tip)
        new_path.set_current_path_nodes(self.current_path_nodes)
        new_path.set_current_path_walked_nodes(self.current_path_walked_nodes.copy())
        return new_path
    
    
     