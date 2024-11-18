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
#from Graph.skeleton_utils import get_skeleton
from Graph.graph_processing import get_pruned_skeleton_graph
from Graph.graph_utils import divide_paths_with_equidistant_nodes
from Graph.visualization import visualize_graph
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\best_models\\best_model_Exp_6_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.5_0.5_8_20.pth'
    dataset_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\test_images'
    skeletons_saving_path = r'C:\Users\maich\Desktop\rootex3\RootEx3.0_GetRsml\skeletons'
    plant_imgs = []
    # Instantiate Predictor with desired thresholds and parameters
    predictor = Predictor(
        model_path,
        device=device,
        resize_height=1400,
        resize_width=1400,
        root_threshold=0.5,
        tip_threshold=0.7,
        source_threshold=0.3,
        sigma=15,            # Custom sigma value
        area_threshold=320,  # Custom area_threshold
        circle_radius=20,    # Custom circle_radius for number_of_circles calculation
        spacing_radius=18    # Custom spacing_radius for spacing calculation
    )

    test_json_files = glob(os.path.join(dataset_path, '*.json'))
    test_dataset = CustomRGBDataset(json_files=test_json_files, image_dir=dataset_path, phase='test', isTraining=False)
    print(f"Number of images in the test dataset: {len(test_dataset)}")
    
    for index in range(len(test_dataset)):
        print(f"\nProcessing image {index+1}/{len(test_dataset)}")
        
        # Predict and visualize the segmentation for the current image in the test dataset
        plant_img = predictor.predict_and_visualize(test_dataset, index)
        
        # Obtain the skeleton from the predicted mask and convert it to an 8-bit unsigned integer image
        skeleton = get_skeleton(plant_img.get_pred_mask()).astype(np.uint8) * 255
        
        # Set the skeleton image in the plant_img object for further processing or visualization
        plant_img.set_skeleton_img(skeleton)
        
        # Save the skeleton image with a filename based on the plant image name
        cv2.imwrite(f'skeleton_{plant_img.get_name()}.png', skeleton)
        
        # Build and prune the skeleton graph, then set it in the plant_img object
        # The graph is saved to a specified path after pruning
        plant_img.set_graph(get_pruned_skeleton_graph(
            plant_img.get_name(),
            skeleton,
            saving_path=f'{skeletons_saving_path}\\pruned_{plant_img.get_name()}.png'
        ))
        
        # Divide paths in the graph by adding equidistant intermediate nodes every 30 pixels
        plant_graph_with_intermed_nodes = divide_paths_with_equidistant_nodes(
            plant_img.get_graph(), 30
        )
        
        # Update the graph in the plant_img object with the new graph containing intermediate nodes
        plant_img.set_graph(plant_graph_with_intermed_nodes)
        
        # Visualize the updated graph overlaid on the skeleton image
        # The visualization is saved to a specified path, and node types are shown
        visualize_graph(
            plant_img.get_skeleton_img(),
            plant_img.get_graph(),
            plant_img.get_name(),
            save_path=f'{skeletons_saving_path}\\intermed_{plant_img.get_name()}.png',
            show_node_types=True
        )
        
        
         


