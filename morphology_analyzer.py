# morphology_analyzer.py
import numpy as np
from cellpose import models, io, utils
from scipy import stats
from skimage import measure, morphology, segmentation
import cv2
from PIL import Image
import pandas as pd
import os

class MorphologyAnalyzer:
    def __init__(self):
        # Try to initialize Cellpose with error handling
        try:
            # Download models if needed
            model_dir = models.MODEL_DIR
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Try to use cyto2 model
            self.model = models.CellposeModel(gpu=False, model_type='cyto2')
            self.cellpose_available = True
        except Exception as e:
            print(f"Cellpose initialization failed: {e}")
            print("Using fallback segmentation method")
            self.model = None
            self.cellpose_available = False
        
        self.pixel_to_um = 0.65  # Default conversion factor
    
    def segment_organoids(self, image_path):
        """Segment organoids using Cellpose or fallback method"""
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Handle uploaded file
            img = Image.open(image_path).convert('L')
            img = np.array(img)
        
        if self.cellpose_available and self.model is not None:
            try:
                # Use Cellpose
                masks, flows, styles = self.model.eval(
                    img, 
                    diameter=None, 
                    channels=[0,0],
                    flow_threshold=0.4,
                    cellprob_threshold=0.0
                )
            except Exception as e:
                print(f"Cellpose segmentation failed: {e}")
                masks = self.fallback_segmentation(img)
        else:
            # Use fallback segmentation
            masks = self.fallback_segmentation(img)
        
        return masks, img
    
    def fallback_segmentation(self, img):
        """Simple watershed segmentation as fallback"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        
        # Threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove small objects
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Find sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that background is 1 instead of 0
        markers = markers + 1
        
        # Mark unknown region as 0
        markers[unknown == 255] = 0
        
        # Apply watershed
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        # Create mask from markers
        masks = markers.copy()
        masks[masks <= 1] = 0
        
        return masks
    
    def measure_organoids(self, masks, min_diameter_um=0):
        """Measure individual organoids"""
        measurements = []
        
        # Get properties of each segmented region
        props = measure.regionprops(masks)
        
        for prop in props:
            # Skip background
            if prop.label == 0:
                continue
                
            # Calculate area and diameter
            area_pixels = prop.area
            area_um2 = area_pixels * (self.pixel_to_um ** 2)
            
            # Equivalent diameter (diameter of circle with same area)
            diameter_pixels = prop.equivalent_diameter
            diameter_um = diameter_pixels * self.pixel_to_um
            
            # Filter by minimum diameter
            if diameter_um >= min_diameter_um:
                measurements.append({
                    'organoid_id': prop.label,
                    'area_pixels': area_pixels,
                    'area_um2': area_um2,
                    'diameter_pixels': diameter_pixels,
                    'diameter_um': diameter_um,
                    'centroid': prop.centroid,
                    'perimeter': prop.perimeter,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity
                })
        
        return pd.DataFrame(measurements)
    
    def compare_conditions(self, control_df, experimental_df):
        """Perform statistical comparison between conditions"""
        results = {}
        
        # Basic statistics
        results['control_stats'] = {
            'count': len(control_df),
            'mean_diameter_um': control_df['diameter_um'].mean() if len(control_df) > 0 else 0,
            'std_diameter_um': control_df['diameter_um'].std() if len(control_df) > 0 else 0,
            'mean_area_um2': control_df['area_um2'].mean() if len(control_df) > 0 else 0,
        }
        
        results['experimental_stats'] = {
            'count': len(experimental_df),
            'mean_diameter_um': experimental_df['diameter_um'].mean() if len(experimental_df) > 0 else 0,
            'std_diameter_um': experimental_df['diameter_um'].std() if len(experimental_df) > 0 else 0,
            'mean_area_um2': experimental_df['area_um2'].mean() if len(experimental_df) > 0 else 0,
        }
        
        # T-test for diameter comparison
        if len(control_df) > 1 and len(experimental_df) > 1:
            t_stat, p_value = stats.ttest_ind(
                control_df['diameter_um'].dropna(), 
                experimental_df['diameter_um'].dropna()
            )
            results['ttest'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            results['ttest'] = {
                't_statistic': None,
                'p_value': None,
                'significant': False
            }
        
        return results
    
    def create_overlay(self, image, masks, measurements_df):
        """Create visualization overlay"""
        # Convert grayscale to RGB for overlay
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = image.copy()
        
        # Create colored masks
        colored_masks = np.zeros_like(overlay)
        
        if len(measurements_df) > 0:
            # Create a color map for organoids
            np.random.seed(42)  # For consistent colors
            for organoid_id in measurements_df['organoid_id'].unique():
                mask = (masks == organoid_id)
                # Generate a distinct color for each organoid
                color = np.random.randint(50, 255, 3)
                colored_masks[mask] = color
        
        # Blend with original
        result = cv2.addWeighted(overlay, 0.7, colored_masks, 0.3, 0)
        
        # Add labels
        for _, row in measurements_df.iterrows():
            cy, cx = map(int, row['centroid'])
            # Add background rectangle for better text visibility
            text = f"D:{row['diameter_um']:.0f}μm"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(result, 
                         (cx - 25, cy - text_height - 2),
                         (cx + text_width - 20, cy + 2),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(result, text, 
                       (cx-20, cy), font, 
                       font_scale, (255, 255, 255), thickness)
        
        return result