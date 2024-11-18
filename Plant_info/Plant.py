class Plant_img:
    #tips = array of tip center coordinates
    #mask image 
    #sources = array of source center coordinates
    skeleton_img = None
    graph = None
    def __init__(self,name,tips,sources,gt_mask,pred_mask,iou,dice,missing_counts, overestimate_counts,gt_sources,gt_tips):
        self.name = name
        self.tips = tips
        self.sources = sources
        self.gt_mask = gt_mask
        self.pred_mask = pred_mask
        self.iou = iou
        self.dice = dice
        self.missing_counts = missing_counts
        self.overestimate_counts = overestimate_counts
        self.gt_sources = gt_sources
        self.gt_tips = gt_tips
        
    def get_graph(self):
        return self.graph
    def set_graph(self,graph):
        self.graph = graph
        
    def get_skeleton_img(self):
        return self.skeleton_img
    def set_skeleton_img(self,skeleton_img):
        self.skeleton_img = skeleton_img
    def get_gt_mask(self):
        return self.gt_mask
    def get_pred_mask(self):
        return self.pred_mask
    def set_gt_mask(self,gt_mask):
        self.gt_mask = gt_mask
    def set_pred_mask(self,pred_mask):
        self.pred_mask = pred_mask
    def get_gt_sources(self):
        return self.gt_sources
    def get_gt_tips(self):
        return self.gt_tips
    def set_gt_sources(self,gt_sources):
        self.gt_sources = gt_sources
    def set_gt_tips(self,gt_tips):
        self.gt_tips = gt_tips
    def get_missing_counts(self):
        return self.missing_counts
    def get_overestimate_counts(self):
        return self.overestimate_counts
    def get_name(self):
        return self.name
    def get_tips(self):
        return self.tips
    def get_sources(self):
        return self.sources
    def get_mask(self):
        return self.mask
    def set_name(self,name):
        self.name = name
    def set_tips(self,tips):
        self.tips = tips
    def set_sources(self,sources):
        self.sources = sources
    def set_mask(self,mask):
        self.mask = mask
    def get_iou(self):
        return self.iou
    def get_dice(self):
        return self.dice
    def get_f1_tip(self):
        return self.f1_tip
    def get_f1_source(self):
        return self.f1_source
    def set_iou(self,iou):
        self.iou = iou
    def set_dice(self,dice):
        self.dice = dice
    def set_overestimate_counts(self,overestimate_counts):
        self.overestimate_counts = overestimate_counts
    def set_missing_counts(self,missing_counts):
        self.missing_counts = missing_counts
        
    def __str__(self):
        return str(self.name) + " " + str(self.tips) + " " + str(self.sources) + " " + str(self.mask) + " " + str(self.iou) + " " + str(self.dice) + " " + str(self.missing_counts) + " " + str(self.overestimate_counts)
    
