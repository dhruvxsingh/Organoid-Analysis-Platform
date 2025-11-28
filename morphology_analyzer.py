"""
Morphology Analyzer for Organoid Counting and Measurement
Version 2.0 - Optimized for organoid detection with tuned parameters
"""

import numpy as np
from scipy import stats, ndimage
from skimage import measure, filters, exposure
from skimage.segmentation import clear_border, watershed
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.feature import peak_local_max
import cv2
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import Cellpose
CELLPOSE_AVAILABLE = False
OMNIPOSE_AVAILABLE = False

try:
    from cellpose import models, core
    CELLPOSE_AVAILABLE = True
    print("✓ Cellpose available")
    
    # Check if Omnipose model is available (Cellpose 3.0+)
    try:
        # Try to check for omnipose
        test_model = models.CellposeModel(model_type='cyto2')
        OMNIPOSE_AVAILABLE = True
        print("✓ Omnipose models available")
    except:
        pass
        
except ImportError as e:
    print(f"Cellpose not available: {e}")
    print("→ Using enhanced OpenCV segmentation")


class MorphologyAnalyzer:
    """
    Analyzer for organoid morphology with accurate counting
    Optimized for brightfield organoid images
    """
    
    def __init__(self, model_type='auto'):
        """
        Initialize the analyzer
        
        Args:
            model_type: 'auto', 'cyto2', 'cyto3', or 'opencv'
        """
        self.model = None
        self.cellpose_available = False
        self.model_type = model_type
        self.pixel_to_um = 0.65  # Default conversion
        
        # Optimized parameters for organoids
        self.params = {
            'diameter': None,  # Will be set based on image or manually
            'flow_threshold': 0.2,  # Lower = detect more (default 0.4)
            'cellprob_threshold': -4,  # Lower = detect fainter objects (default 0)
            'min_size': 300,  # Minimum object size in pixels
        }
        
        self._initialize_model(model_type)
    
    def _initialize_model(self, model_type):
        """Initialize the segmentation model"""
        if not CELLPOSE_AVAILABLE or model_type == 'opencv':
            print("→ Using OpenCV-based segmentation")
            self.cellpose_available = False
            return
        
        try:
            # Try different models in order of preference for organoids
            models_to_try = []
            
            if model_type == 'auto':
                # Order: cyto3 (newest) > cyto2 > cyto
                models_to_try = ['cyto3', 'cyto2', 'cyto']
            else:
                models_to_try = [model_type]
            
            for m_type in models_to_try:
                try:
                    self.model = models.CellposeModel(gpu=False, model_type=m_type)
                    self.cellpose_available = True
                    self.model_type = m_type
                    print(f"✓ Loaded Cellpose model: {m_type}")
                    break
                except Exception as e:
                    print(f"  Could not load {m_type}: {e}")
                    continue
            
            if not self.cellpose_available:
                print("→ Falling back to OpenCV segmentation")
                
        except Exception as e:
            print(f"Model initialization failed: {e}")
            print("→ Using OpenCV segmentation")
            self.cellpose_available = False
    
    def set_parameters(self, diameter=None, flow_threshold=0.2, 
                       cellprob_threshold=-4, min_size=300):
        """
        Set segmentation parameters
        
        Args:
            diameter: Expected organoid diameter in pixels (None = auto)
            flow_threshold: 0.1-0.5, lower = more objects detected
            cellprob_threshold: -6 to 6, lower = detect fainter objects
            min_size: Minimum object size in pixels
        """
        self.params['diameter'] = diameter
        self.params['flow_threshold'] = flow_threshold
        self.params['cellprob_threshold'] = cellprob_threshold
        self.params['min_size'] = min_size
        
        print(f"Parameters set: diameter={diameter}, flow={flow_threshold}, "
              f"cellprob={cellprob_threshold}, min_size={min_size}")
    
    def load_image(self, image_input):
        """Load image from file path or uploaded file object"""
        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    pil_img = Image.open(image_input).convert('L')
                    img = np.array(pil_img)
            else:
                image_input.seek(0)
                pil_img = Image.open(image_input).convert('L')
                img = np.array(pil_img)
            
            if img is None or img.size == 0:
                raise ValueError("Failed to load image")
            
            return img
            
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    def preprocess_image(self, img, enhance_contrast=True, denoise=True):
        """
        Advanced preprocessing for organoid images
        
        Steps:
        1. Normalize intensity
        2. CLAHE contrast enhancement
        3. Gentle denoising
        4. Background correction
        """
        # Normalize to 0-255
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Check if inversion needed (dark organoids on light background)
        h, w = img_norm.shape
        border_vals = np.concatenate([
            img_norm[:10, :].flatten(),
            img_norm[-10:, :].flatten(),
            img_norm[:, :10].flatten(),
            img_norm[:, -10:].flatten()
        ])
        border_mean = np.mean(border_vals)
        center_mean = img_norm[h//4:3*h//4, w//4:3*w//4].mean()
        
        needs_inversion = border_mean > center_mean + 5
        
        if needs_inversion:
            img_proc = 255 - img_norm
        else:
            img_proc = img_norm
        
        # Denoise (gentle Gaussian blur)
        if denoise:
            img_proc = cv2.GaussianBlur(img_proc, (3, 3), 1.0)
        
        # CLAHE contrast enhancement
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img_proc = clahe.apply(img_proc)
        
        # Background subtraction (rolling ball approximation)
        background = cv2.GaussianBlur(img_proc, (51, 51), 0)
        img_proc = cv2.subtract(img_proc, background // 2)
        img_proc = cv2.normalize(img_proc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return img_proc, img_norm, needs_inversion
    
    def estimate_diameter(self, img):
        """
        Estimate typical organoid diameter from image
        Uses initial thresholding to find objects
        """
        # Quick threshold
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 100  # Default
        
        # Calculate equivalent diameters
        diameters = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Skip tiny objects
                diameter = np.sqrt(4 * area / np.pi)
                diameters.append(diameter)
        
        if len(diameters) == 0:
            return 100
        
        # Use median diameter
        estimated = np.median(diameters)
        print(f"  Estimated diameter: {estimated:.0f} pixels")
        
        return estimated
    
    def segment_organoids(self, image_input, min_size=None, diameter=None,
                          use_cellpose=True, enhance_preprocessing=True):
        """
        Main segmentation function with optimized parameters for organoids
        
        Args:
            image_input: File path or uploaded file
            min_size: Minimum object size (uses self.params if None)
            diameter: Expected diameter (uses self.params or auto if None)
            use_cellpose: Try Cellpose first
            enhance_preprocessing: Apply enhanced preprocessing
        
        Returns:
            masks: Labeled mask (each organoid has unique ID)
            original_img: Original image for display
        """
        # Load image
        original_img = self.load_image(image_input)
        
        # Use parameters
        if min_size is None:
            min_size = self.params['min_size']
        
        # Preprocess
        processed, normalized, was_inverted = self.preprocess_image(
            original_img, 
            enhance_contrast=enhance_preprocessing,
            denoise=enhance_preprocessing
        )
        
        # Estimate diameter if not provided
        if diameter is None and self.params['diameter'] is None:
            diameter = self.estimate_diameter(processed)
        elif diameter is None:
            diameter = self.params['diameter']
        
        masks = None
        method_used = ""
        
        # Try Cellpose with optimized parameters
        if use_cellpose and self.cellpose_available and self.model is not None:
            try:
                print(f"  Running Cellpose ({self.model_type}) with:")
                print(f"    diameter={diameter}, flow={self.params['flow_threshold']}, "
                      f"cellprob={self.params['cellprob_threshold']}")
                
                masks, flows, styles = self.model.eval(
                    processed,
                    diameter=diameter,
                    channels=[0, 0],
                    flow_threshold=self.params['flow_threshold'],
                    cellprob_threshold=self.params['cellprob_threshold'],
                    min_size=min_size // 4,  # Cellpose min_size is different
                )
                
                if masks is not None and masks.max() > 0:
                    method_used = f"Cellpose ({self.model_type})"
                    print(f"  Cellpose found {masks.max()} objects")
                else:
                    masks = None
                    print("  Cellpose found 0 objects, trying fallback...")
                    
            except Exception as e:
                print(f"  Cellpose error: {e}")
                masks = None
        
        # Fallback to OpenCV
        if masks is None or masks.max() == 0:
            print("  Using OpenCV watershed segmentation...")
            masks = self.segment_opencv_enhanced(processed, min_size, diameter)
            method_used = "OpenCV Watershed"
        
        # Post-processing
        masks = self.postprocess_masks(masks, min_size)
        
        final_count = masks.max() if masks is not None else 0
        print(f"  Final: {final_count} organoids ({method_used})")
        
        return masks, original_img
    
    def segment_opencv_enhanced(self, img, min_size=300, expected_diameter=None):
        """
        Enhanced OpenCV segmentation optimized for organoids
        Uses multiple techniques and combines results
        """
        h, w = img.shape
        
        # Method 1: Adaptive thresholding
        block_size = max(31, int(expected_diameter * 0.8) if expected_diameter else 51)
        if block_size % 2 == 0:
            block_size += 1
        
        adaptive = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=-3
        )
        
        # Method 2: Otsu thresholding
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 3: Multi-level thresholding
        thresh_vals = [
            np.percentile(img, 60),
            np.percentile(img, 70),
            np.percentile(img, 80)
        ]
        multi_thresh = np.zeros_like(img)
        for t in thresh_vals:
            _, binary = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
            multi_thresh = cv2.bitwise_or(multi_thresh, binary)
        
        # Combine all methods (voting)
        combined = np.zeros_like(img, dtype=np.float32)
        combined += (adaptive > 0).astype(np.float32)
        combined += (otsu > 0).astype(np.float32)
        combined += (multi_thresh > 0).astype(np.float32)
        
        # Require at least 2 methods to agree
        final_binary = (combined >= 2).astype(np.uint8) * 255
        
        # Morphological cleanup
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Open to remove noise
        cleaned = cv2.morphologyEx(final_binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Close to fill holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
        
        # Remove border-touching objects
        cleaned = clear_border(cleaned)
        
        # Distance transform for watershed seeds
        dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
        
        # Find local maxima as seeds
        # Adaptive threshold based on expected size
        if expected_diameter:
            min_distance = int(expected_diameter * 0.3)
        else:
            min_distance = 20
        
        local_max = peak_local_max(
            dist,
            min_distance=min_distance,
            threshold_abs=0.2,
            exclude_border=False
        )
        
        # Create markers from local maxima
        markers = np.zeros_like(cleaned, dtype=np.int32)
        for idx, (y, x) in enumerate(local_max, start=1):
            markers[y, x] = idx
        
        # Dilate markers slightly
        markers = cv2.dilate(markers.astype(np.uint8), kernel_small, iterations=2).astype(np.int32)
        
        # Watershed
        # Need 3-channel image for watershed
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Mark background
        markers[cleaned == 0] = -1
        
        cv2.watershed(img_3ch, markers)
        
        # Clean up markers
        masks = markers.copy()
        masks[masks <= 0] = 0
        
        return masks
    
    def postprocess_masks(self, masks, min_size=300):
        """
        Post-process masks to clean up segmentation
        """
        if masks is None or masks.max() == 0:
            return np.zeros_like(masks) if masks is not None else None
        
        # Remove small objects
        cleaned = np.zeros_like(masks)
        
        for label_id in range(1, masks.max() + 1):
            obj_mask = (masks == label_id)
            area = np.sum(obj_mask)
            
            if area >= min_size:
                cleaned[obj_mask] = label_id
        
        # Relabel sequentially
        unique_labels = sorted([l for l in np.unique(cleaned) if l > 0])
        relabeled = np.zeros_like(cleaned)
        
        for new_id, old_id in enumerate(unique_labels, start=1):
            relabeled[cleaned == old_id] = new_id
        
        return relabeled
    
    def segment_with_watershed_split(self, masks, original_img, min_size=300):
        """
        Additional step to split merged organoids using watershed
        Useful when Cellpose merges touching organoids
        """
        if masks is None or masks.max() == 0:
            return masks
        
        new_masks = np.zeros_like(masks)
        next_label = 1
        
        for label_id in range(1, masks.max() + 1):
            obj_mask = (masks == label_id).astype(np.uint8)
            area = np.sum(obj_mask)
            
            # Check if object might be merged (very large or non-circular)
            props = measure.regionprops(obj_mask.astype(int))
            if len(props) == 0:
                continue
            
            prop = props[0]
            circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 1
            
            # If object is large and non-circular, try to split
            should_split = (area > min_size * 4) or (circularity < 0.5 and area > min_size * 2)
            
            if should_split:
                # Use distance transform + watershed to split
                dist = cv2.distanceTransform(obj_mask, cv2.DIST_L2, 5)
                
                # Find peaks
                local_max = peak_local_max(dist, min_distance=20, threshold_abs=0.3 * dist.max())
                
                if len(local_max) > 1:
                    # Multiple peaks = try to split
                    markers = np.zeros_like(obj_mask, dtype=np.int32)
                    for i, (y, x) in enumerate(local_max, start=1):
                        markers[y, x] = i
                    
                    markers = cv2.dilate(markers.astype(np.uint8), 
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                                        iterations=2).astype(np.int32)
                    
                    markers[obj_mask == 0] = -1
                    
                    img_3ch = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
                    cv2.watershed(img_3ch, markers)
                    
                    for split_id in range(1, markers.max() + 1):
                        split_mask = (markers == split_id)
                        if np.sum(split_mask) >= min_size:
                            new_masks[split_mask] = next_label
                            next_label += 1
                else:
                    # Single peak, keep as is
                    new_masks[obj_mask > 0] = next_label
                    next_label += 1
            else:
                # Normal object, keep as is
                new_masks[obj_mask > 0] = next_label
                next_label += 1
        
        return new_masks
    
    def measure_organoids(self, masks, min_diameter_um=0):
        """Measure properties of each segmented organoid"""
        if masks is None or masks.max() == 0:
            return pd.DataFrame()
        
        measurements = []
        props = measure.regionprops(masks)
        
        for prop in props:
            if prop.label == 0:
                continue
            
            area_px = prop.area
            area_um2 = area_px * (self.pixel_to_um ** 2)
            
            diameter_px = prop.equivalent_diameter
            diameter_um = diameter_px * self.pixel_to_um
            
            perimeter_px = prop.perimeter
            perimeter_um = perimeter_px * self.pixel_to_um
            
            circularity = (4 * np.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else 0
            
            if diameter_um >= min_diameter_um:
                measurements.append({
                    'organoid_id': int(prop.label),
                    'area_pixels': int(area_px),
                    'area_um2': round(area_um2, 2),
                    'diameter_pixels': round(diameter_px, 2),
                    'diameter_um': round(diameter_um, 2),
                    'perimeter_pixels': round(perimeter_px, 2),
                    'perimeter_um': round(perimeter_um, 2),
                    'centroid': prop.centroid,
                    'bbox': prop.bbox,
                    'eccentricity': round(prop.eccentricity, 3),
                    'solidity': round(prop.solidity, 3),
                    'circularity': round(min(circularity, 1.0), 3)
                })
        
        df = pd.DataFrame(measurements)
        if len(df) > 0:
            df = df.sort_values('organoid_id').reset_index(drop=True)
        
        return df
    
    def compare_conditions(self, control_df, experimental_df):
        """Statistical comparison between control and experimental groups"""
        results = {}
        
        # Control stats
        if len(control_df) > 0:
            results['control_stats'] = {
                'count': len(control_df),
                'mean_diameter_um': round(control_df['diameter_um'].mean(), 2),
                'std_diameter_um': round(control_df['diameter_um'].std(), 2),
                'median_diameter_um': round(control_df['diameter_um'].median(), 2),
                'min_diameter_um': round(control_df['diameter_um'].min(), 2),
                'max_diameter_um': round(control_df['diameter_um'].max(), 2),
                'mean_area_um2': round(control_df['area_um2'].mean(), 2),
                'mean_circularity': round(control_df['circularity'].mean(), 3),
            }
        else:
            results['control_stats'] = {
                'count': 0, 'mean_diameter_um': 0, 'std_diameter_um': 0,
                'median_diameter_um': 0, 'min_diameter_um': 0, 'max_diameter_um': 0,
                'mean_area_um2': 0, 'mean_circularity': 0,
            }
        
        # Experimental stats
        if len(experimental_df) > 0:
            results['experimental_stats'] = {
                'count': len(experimental_df),
                'mean_diameter_um': round(experimental_df['diameter_um'].mean(), 2),
                'std_diameter_um': round(experimental_df['diameter_um'].std(), 2),
                'median_diameter_um': round(experimental_df['diameter_um'].median(), 2),
                'min_diameter_um': round(experimental_df['diameter_um'].min(), 2),
                'max_diameter_um': round(experimental_df['diameter_um'].max(), 2),
                'mean_area_um2': round(experimental_df['area_um2'].mean(), 2),
                'mean_circularity': round(experimental_df['circularity'].mean(), 3),
            }
        else:
            results['experimental_stats'] = {
                'count': 0, 'mean_diameter_um': 0, 'std_diameter_um': 0,
                'median_diameter_um': 0, 'min_diameter_um': 0, 'max_diameter_um': 0,
                'mean_area_um2': 0, 'mean_circularity': 0,
            }
        
        # T-test
        if len(control_df) > 1 and len(experimental_df) > 1:
            t_stat, p_value = stats.ttest_ind(
                control_df['diameter_um'].dropna(),
                experimental_df['diameter_um'].dropna(),
                equal_var=False
            )
            results['ttest'] = {
                't_statistic': round(t_stat, 4),
                'p_value': round(p_value, 6),
                'significant': p_value < 0.05
            }
        else:
            results['ttest'] = {'t_statistic': None, 'p_value': None, 'significant': False}
        
        return results
    
    def create_overlay(self, image, masks, measurements_df,
                       show_labels=True, show_contours=True,
                       contour_thickness=2, label_type='both'):
        """Create visualization overlay with boundaries"""
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = image.copy()
        
        overlay = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if masks is None or len(measurements_df) == 0:
            cv2.putText(overlay, "No organoids detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return overlay
        
        num_objects = len(measurements_df)
        colors = self._generate_colors(num_objects)
        
        for idx, (_, row) in enumerate(measurements_df.iterrows()):
            organoid_id = int(row['organoid_id'])
            color = colors[idx]
            
            object_mask = (masks == organoid_id).astype(np.uint8)
            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if show_contours and contours:
                # Semi-transparent fill
                mask_colored = np.zeros_like(overlay)
                cv2.drawContours(mask_colored, contours, -1, color, -1)
                overlay = cv2.addWeighted(overlay, 1, mask_colored, 0.3, 0)
                
                # Bold contour
                cv2.drawContours(overlay, contours, -1, color, contour_thickness)
            
            if show_labels:
                cy, cx = int(row['centroid'][0]), int(row['centroid'][1])
                
                if label_type == 'id':
                    text = f"#{organoid_id}"
                elif label_type == 'diameter':
                    text = f"{row['diameter_um']:.0f}um"
                else:
                    text = f"#{organoid_id}: {row['diameter_um']:.0f}um"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                text_x = max(5, min(cx - text_w // 2, overlay.shape[1] - text_w - 5))
                text_y = max(text_h + 5, min(cy, overlay.shape[0] - 5))
                
                cv2.rectangle(overlay, (text_x - 2, text_y - text_h - 2),
                             (text_x + text_w + 2, text_y + 2), (0, 0, 0), -1)
                cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Count in corner
        cv2.putText(overlay, f"Count: {num_objects}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay
    
    def create_boundary_overlay(self, image, masks, measurements_df):
        """Create clean overlay with only boundaries (no fill)"""
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = image.copy()
        
        overlay = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if masks is None or len(measurements_df) == 0:
            return overlay
        
        colors = self._generate_colors(len(measurements_df))
        
        for idx, (_, row) in enumerate(measurements_df.iterrows()):
            organoid_id = int(row['organoid_id'])
            color = colors[idx]
            
            object_mask = (masks == organoid_id).astype(np.uint8)
            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Thick boundary only
            cv2.drawContours(overlay, contours, -1, color, 3)
            
            # Small label circle
            cy, cx = int(row['centroid'][0]), int(row['centroid'][1])
            cv2.circle(overlay, (cx, cy), 12, (0, 0, 0), -1)
            cv2.circle(overlay, (cx, cy), 12, color, 2)
            
            text = str(organoid_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
            cv2.putText(overlay, text, (cx - tw//2, cy + th//2), font, 0.4, (255, 255, 255), 1)
        
        cv2.putText(overlay, f"Count: {len(measurements_df)}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay
    
    def _generate_colors(self, n):
        """Generate n visually distinct colors"""
        if n == 0:
            return []
        
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr)))
        
        return colors


# ============ TESTING FUNCTION ============

def test_with_parameters(image_path, diameter=None, flow_threshold=0.2, 
                         cellprob_threshold=-4, min_size=300):
    """
    Test segmentation with specific parameters
    Useful for finding optimal settings for your images
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {image_path}")
    print(f"{'='*60}")
    print(f"Parameters: diameter={diameter}, flow={flow_threshold}, "
          f"cellprob={cellprob_threshold}, min_size={min_size}")
    print()
    
    analyzer = MorphologyAnalyzer()
    analyzer.set_parameters(
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size
    )
    
    masks, original = analyzer.segment_organoids(image_path)
    measurements = analyzer.measure_organoids(masks)
    
    print(f"\n{'='*60}")
    print(f"RESULT: {len(measurements)} organoids detected")
    print(f"{'='*60}")
    
    if len(measurements) > 0:
        print("\nMeasurements:")
        print(measurements[['organoid_id', 'diameter_um', 'area_um2', 'circularity']].to_string())
        
        # Save overlays
        overlay = analyzer.create_overlay(original, masks, measurements)
        boundary = analyzer.create_boundary_overlay(original, masks, measurements)
        
        base_name = image_path.rsplit('.', 1)[0]
        cv2.imwrite(f"{base_name}_overlay.png", overlay)
        cv2.imwrite(f"{base_name}_boundary.png", boundary)
        
        print(f"\n✓ Saved: {base_name}_overlay.png")
        print(f"✓ Saved: {base_name}_boundary.png")
    
    return measurements


def parameter_sweep(image_path):
    """
    Try multiple parameter combinations to find the best settings
    """
    print(f"\n{'='*60}")
    print(f"PARAMETER SWEEP: {image_path}")
    print(f"{'='*60}\n")
    
    # Parameters to try
    diameters = [None, 50, 80, 100, 120]
    flow_thresholds = [0.1, 0.2, 0.3, 0.4]
    cellprob_thresholds = [-6, -4, -2, 0]
    
    results = []
    
    analyzer = MorphologyAnalyzer()
    original = analyzer.load_image(image_path)
    
    for diameter in diameters:
        for flow in flow_thresholds:
            for cellprob in cellprob_thresholds:
                analyzer.set_parameters(
                    diameter=diameter,
                    flow_threshold=flow,
                    cellprob_threshold=cellprob
                )
                
                try:
                    masks, _ = analyzer.segment_organoids(image_path)
                    count = masks.max() if masks is not None else 0
                    
                    results.append({
                        'diameter': diameter,
                        'flow_threshold': flow,
                        'cellprob_threshold': cellprob,
                        'count': count
                    })
                    
                    print(f"d={diameter}, flow={flow}, cellprob={cellprob} → {count} objects")
                    
                except Exception as e:
                    print(f"Error: {e}")
    
    # Show summary
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("PARAMETER SWEEP RESULTS")
    print(f"{'='*60}")
    print(df.to_string())
    
    # Find most common count (likely correct)
    from collections import Counter
    counts = Counter(df['count'])
    print(f"\nCount frequency: {dict(counts)}")
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if len(sys.argv) > 2 and sys.argv[2] == '--sweep':
            # Run parameter sweep
            parameter_sweep(image_path)
        else:
            # Normal test with optional parameters
            diameter = int(sys.argv[2]) if len(sys.argv) > 2 else None
            min_size = int(sys.argv[3]) if len(sys.argv) > 3 else 300
            
            test_with_parameters(
                image_path,
                diameter=diameter,
                min_size=min_size
            )
    else:
        print("Usage:")
        print("  python morphology_analyzer.py <image_path> [diameter] [min_size]")
        print("  python morphology_analyzer.py <image_path> --sweep  (parameter sweep)")
        print()
        print("Examples:")
        print("  python morphology_analyzer.py organoid.tif")
        print("  python morphology_analyzer.py organoid.tif 100 500")
        print("  python morphology_analyzer.py organoid.tif --sweep")