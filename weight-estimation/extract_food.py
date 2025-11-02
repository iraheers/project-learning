import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# this script extracts color images of each component except background using SAM masks

def extract_color_components(image_path, masks_dir, labels_csv, output_dir):
    """
    Extract color images of each component except background using SAM masks
    
    Args:
        image_path: Path to original input image
        masks_dir: Directory containing SAM mask images
        labels_csv: Path to the CSV file with mask labels
        output_dir: Directory to save color component images
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Read labels CSV
    labels_df = pd.read_csv(labels_csv)
    
    # Filter out background components (category_id == 0)
    non_background_df = labels_df[labels_df['category_id'] != 0]
    
    print(f"Found {len(non_background_df)} non-background components")
    
    for _, row in non_background_df.iterrows():
        mask_id = row['id']
        category_name = row['category_name']
        
        # Construct mask file path (assuming masks are named like mask_0.png, mask_1.png, etc.)
        mask_path = os.path.join(masks_dir, f"{mask_id}.png")
        
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            continue
        
        # Read the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Could not load mask: {mask_path}")
            continue
        
        # Create a colored version of the component
        color_component = np.zeros_like(original_image)
        
        # Apply mask to extract the component from original image
        color_component = cv2.bitwise_and(original_image, original_image, mask=mask)
        
        # Create output filename
        output_filename = f"{category_name}_{mask_id}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the color component
        cv2.imwrite(output_path, color_component)
        print(f"Saved: {output_filename}")
    
    print(f"Extracted {len(non_background_df)} color components to {output_dir}")

def extract_with_white_background(image_path, masks_dir, labels_csv, output_dir):
    """
    Extract components with white background instead of black
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Read labels CSV
    labels_df = pd.read_csv(labels_csv)
    
    # Filter out background components
    non_background_df = labels_df[labels_df['category_id'] != 0]
    
    # Create combined mask (all non-background components)
    combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    
    for _, row in non_background_df.iterrows():
        mask_id = row['id']
        category_name = row['category_name']
        
        mask_path = os.path.join(masks_dir, f"{mask_id}.png")
        
        if not os.path.exists(mask_path):
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Add to combined mask
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Create white background for individual component
        white_bg = np.ones_like(original_image) * 255
        
        # Extract component using mask
        component = cv2.bitwise_and(original_image, original_image, mask=mask)
        
        # Invert mask for background
        mask_inv = cv2.bitwise_not(mask)
        
        # Combine component with white background for individual image
        white_bg = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
        result = cv2.add(component, white_bg)
        
        # Save individual component
        output_filename = f"{category_name}_{mask_id}.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_filename}")
    
    # Create combined image using the combined mask
    combined_image = cv2.bitwise_and(original_image, original_image, mask=combined_mask)
    
    # Save combined image with white background
    white_bg_combined = np.ones_like(original_image) * 255
    combined_mask_inv = cv2.bitwise_not(combined_mask)
    white_bg_combined = cv2.bitwise_and(white_bg_combined, white_bg_combined, mask=combined_mask_inv)
    final_combined = cv2.add(combined_image, white_bg_combined)
    
    combined_output_path = os.path.join(output_dir, "_meal.png")
    cv2.imwrite(combined_output_path, final_combined)
    print(f"Saved combined image: {combined_output_path}")

# Usage example
if __name__ == "__main__":
    # Set your paths here
    image = "real-image"  # Example image name
    input_image = f"FoodSAM/outputs/{image}/input.jpg"  # Your original image
    sam_masks_dir = f"FoodSAM/outputs/{image}/sam_mask"  # Folder with mask_0.png, mask_1.png, etc.
    labels_file = f"FoodSAM/outputs/{image}/sam_mask_label/sam_mask_label.txt"  # Your CSV file
    #output_directory = f"FoodSAM/output_components/{image}"  # Where to save results

    # Extract components with transparent/black background
    #extract_color_components(input_image, sam_masks_dir, labels_file, output_directory)
    
    # Optional: Extract with white background
    white_bg_dir = f"InstantMesh/segments/{image}"
    extract_with_white_background(input_image, sam_masks_dir, labels_file, white_bg_dir)