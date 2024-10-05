import cv2
import numpy as np
import os

def create_bounding_box_from_mask(mask_path):
    # Load the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find the coordinates of non-zero pixels
    coords = cv2.findNonZero(mask)
    
    # Get the bounding box from the mask
    x, y, w, h = cv2.boundingRect(coords)
    
    # Calculate the center, width, and height in normalized coordinates
    image_height, image_width = mask.shape[:2]
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    width = w / image_width
    height = h / image_height
    
    return x_center, y_center, width, height

def process_masks_and_save_labels(mask_dir, label_dir, class_id=0):
    os.makedirs(label_dir, exist_ok=True)
    
    for mask_filename in os.listdir(mask_dir):
        if mask_filename.endswith('.png') or mask_filename.endswith('.jpg'):
            mask_path = os.path.join(mask_dir, mask_filename)
            x_center, y_center, width, height = create_bounding_box_from_mask(mask_path)
            
            # Create the corresponding label file
            label_filename = os.path.splitext(mask_filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)
            
            # Write the bounding box in YOLO format
            with open(label_path, 'w') as label_file:
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Example usage
mask_directory = 'mask'   # Directory containing the segmentation masks
label_directory = 'labels'  # Directory to save the YOLO labels
process_masks_and_save_labels(mask_directory, label_directory)
