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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'best_models\\best_model_Exp_6_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.5_0.5_8_20.pth'
    dataset_path = 'test_images'
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
        plant_imgs.append(predictor.predict_and_visualize(predictor, test_dataset, index))
    
    print('done')

