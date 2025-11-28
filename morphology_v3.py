"""
Organoid Analyzer v3.0
Specialized for organoid detection with multiple model options
"""

import numpy as np
from scipy import stats, ndimage
from skimage import measure, filters, exposure, segmentation
from skimage.morphology import remove_small_objects, remove_small_holes, disk, white_tophat
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border
import cv2
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Check available backends
CELLPOSE_AVAILABLE = False
STARDIST_AVAILABLE = False

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
    print("✓ Cellpose available")
except ImportError:
    print("✗ Cellpose not available")

try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    STARDIST_AVAILABLE = True
    print("✓ StarDist available")
except ImportError:
    print("✗ StarDist not available (install with: pip install stardist)")


class OrganoidAnalyzer:
    """
    Specialized analyzer for organoid detection
    Supports multiple segmentation backends
    """
    
    def __init__(self, method='auto'):
        """
        Initialize analyzer
        
        Args:
            method: 'auto', 'stardist', 'cellpose', 'classical', or 'threshold'
        """
        self.method = method
        self.pixel_to_um = 0.65
        self.model = None
        self.model_type = None
        
        # Detection parameters optimized for organoids
        self.params = {
            'min_size_pixels': 500,      # Minimum organoid size
            'max_size_pixels': 100000,   # Maximum (filter out huge artifacts)
            'min_circularity': 0.3,      # Organoids are somewhat round
            'intensity_threshold': 'auto',
            'expected_diameter': None,    # Will be estimated
        }
        
        self._initialize_backend(method)
    
    def _initialize_backend(self, method):
        """Initialize the segmentation backend"""
        
        if method == 'auto':
            # Priority: StarDist > Classical > Cellpose for organoids
            if STARDIST_AVAILABLE:
                self._init_stardist()
            else:
                print("→ Using Classical CV (optimized for organoids)")
                self.model_type = 'classical'
        
        elif method == 'stardist':
            if STARDIST_AVAILABLE:
                self._init_stardist()
            else:
                raise ImportError("StarDist not installed. Run: pip install stardist tensorflow")
        
        elif method == 'cellpose':
            if CELLPOSE_AVAILABLE:
                self._init_cellpose()
            else:
                raise ImportError("Cellpose not installed")
        
        elif method in ['classical', 'threshold']:
            self.model_type = method
            print(f"→ Using {method} segmentation")
        
        else:
            self.model_type = 'classical'
    
    def _init_stardist(self):
        """Initialize StarDist with versatile fluorescent nuclei model"""
        try:
            # '2D_versatile_fluo' works well for round objects
            self.model = StarDist2D.from_pretrained('2D_versatile_fluo')
            self.model_type = 'stardist'
            print("✓ StarDist model loaded (2D_versatile_fluo)")
        except Exception as e:
            print(f"StarDist init failed: {e}")
            self.model_type = 'classical'
    
    def _init_cellpose(self):
        """Initialize Cellpose"""
        try:
            self.model = models.CellposeModel(gpu=False, model_type='cyto2')
            self.model_type = 'cellpose'
            print("✓ Cellpose model loaded")
        except Exception as e:
            print(f"Cellpose init failed: {e}")
            self.model_type = 'classical'
    
    def load_image(self, image_input):
        """Load image from path or file object"""
        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = np.array(Image.open(image_input).convert('L'))
            else:
                image_input.seek(0)
                img = np.array(Image.open(image_input).convert('L'))
            
            return img
        except Exception as e:
            raise ValueError(f"Cannot load image: {e}")
    
    def preprocess_for_organoids(self, img):
        """
        Preprocessing specifically optimized for brightfield organoid images
        """
        # Step 1: Normalize
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Step 2: Determine if organoids are dark or bright
        # Sample border vs center
        h, w = img_norm.shape
        border = np.mean([
            img_norm[:20, :].mean(),
            img_norm[-20:, :].mean(),
            img_norm[:, :20].mean(),
            img_norm[:, -20:].mean()
        ])
        center = img_norm[h//4:3*h//4, w//4:3*w//4].mean()
        
        # Organoids typically appear darker in brightfield
        if border > center:
            # Dark organoids on light background - INVERT
            img_inv = 255 - img_norm
        else:
            img_inv = img_norm
        
        # Step 3: Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        img_clahe = clahe.apply(img_inv)
        
        # Step 4: Reduce noise while preserving edges
        img_bilateral = cv2.bilateralFilter(img_clahe, 9, 75, 75)
        
        # Step 5: Background subtraction (removes uneven illumination)
        # Use large kernel for background estimation
        kernel_size = max(img.shape) // 4
        if kernel_size % 2 == 0:
            kernel_size += 1
        background = cv2.GaussianBlur(img_bilateral, (kernel_size, kernel_size), 0)
        img_corrected = cv2.subtract(img_bilateral, background // 2)
        img_corrected = cv2.normalize(img_corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return img_corrected, img_norm
    
    def segment_organoids(self, image_input, min_size=None):
        """
        Main segmentation function
        
        Returns:
            masks: Labeled mask array
            original: Original image
        """
        # Load
        original = self.load_image(image_input)
        
        # Preprocess
        processed, normalized = self.preprocess_for_organoids(original)
        
        # Get min_size
        if min_size is None:
            min_size = self.params['min_size_pixels']
        
        # Segment based on backend
        if self.model_type == 'stardist' and self.model is not None:
            masks = self._segment_stardist(processed)
        elif self.model_type == 'cellpose' and self.model is not None:
            masks = self._segment_cellpose(processed)
        else:
            masks = self._segment_classical(processed, original)
        
        # Post-process
        masks = self._postprocess(masks, min_size)
        
        print(f"Detected {masks.max()} organoids using {self.model_type}")
        
        return masks, original
    
    def _segment_stardist(self, img):
        """Segment using StarDist"""
        # Normalize for StarDist
        img_normalized = normalize(img, 1, 99.8)
        
        # Predict
        labels, details = self.model.predict_instances(img_normalized)
        
        return labels
    
    def _segment_cellpose(self, img):
        """Segment using Cellpose with organoid-optimized parameters"""
        masks, _, _ = self.model.eval(
            img,
            diameter=self.params.get('expected_diameter', None),
            channels=[0, 0],
            flow_threshold=0.1,      # Very low - detect more
            cellprob_threshold=-6,   # Very low - detect faint objects
        )
        return masks
    
    def _segment_classical(self, processed, original):
        """
        Classical computer vision approach optimized for organoids
        This often works BETTER than deep learning for brightfield organoids!
        """
        h, w = processed.shape
        
        # === STEP 1: Multi-scale detection ===
        # Organoids can vary in size, so we use multiple approaches
        
        # Method A: Adaptive thresholding (good for varying illumination)
        block_size = max(51, min(h, w) // 10)
        if block_size % 2 == 0:
            block_size += 1
        
        adaptive = cv2.adaptiveThreshold(
            processed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, -5
        )
        
        # Method B: Otsu thresholding
        _, otsu = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method C: Edge-based detection
        edges = cv2.Canny(processed, 30, 100)
        edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        
        # Fill the edges to get objects
        edges_filled = ndimage.binary_fill_holes(edges_dilated).astype(np.uint8) * 255
        
        # === STEP 2: Combine methods ===
        # Use voting - pixel is foreground if at least 2 methods agree
        combined = np.zeros_like(processed, dtype=np.float32)
        combined += (adaptive > 0).astype(np.float32)
        combined += (otsu > 0).astype(np.float32)
        combined += (edges_filled > 0).astype(np.float32)
        
        binary = (combined >= 2).astype(np.uint8) * 255
        
        # === STEP 3: Morphological cleanup ===
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fill holes in organoids
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Fill any remaining holes
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255
        
        # === STEP 4: Separate touching organoids with watershed ===
        # Distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
        
        # Find local maxima (organoid centers)
        # Use adaptive min_distance based on image size
        min_distance = max(20, min(h, w) // 30)
        
        coordinates = peak_local_max(
            dist,
            min_distance=min_distance,
            threshold_abs=0.15,
            exclude_border=False
        )
        
        # Create markers
        markers = np.zeros(binary.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coordinates, start=1):
            markers[y, x] = i
        
        # Expand markers slightly
        markers = cv2.dilate(
            markers.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=3
        ).astype(np.int32)
        
        # Background marker
        sure_bg = cv2.dilate(binary, kernel, iterations=5)
        markers[sure_bg == 0] = -1  # Background
        markers[binary == 0] = -1   # Also mark holes as background
        
        # Watershed
        img_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img_color, markers)
        
        # Create output mask
        masks = markers.copy()
        masks[masks <= 0] = 0
        
        return masks
    
    def _postprocess(self, masks, min_size):
        """Post-process masks: filter by size and shape"""
        if masks is None or masks.max() == 0:
            return np.zeros_like(masks) if masks is not None else None
        
        output = np.zeros_like(masks)
        new_label = 1
        
        props = measure.regionprops(masks)
        
        for prop in props:
            # Size filter
            if prop.area < min_size:
                continue
            if prop.area > self.params['max_size_pixels']:
                continue
            
            # Shape filter (circularity)
            if prop.perimeter > 0:
                circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2)
                if circularity < self.params['min_circularity']:
                    continue
            
            # Keep this object
            output[masks == prop.label] = new_label
            new_label += 1
        
        return output
    
    def measure_organoids(self, masks, min_diameter_um=0):
        """Measure properties of detected organoids"""
        if masks is None or masks.max() == 0:
            return pd.DataFrame()
        
        measurements = []
        props = measure.regionprops(masks)
        
        for prop in props:
            area_px = prop.area
            area_um2 = area_px * (self.pixel_to_um ** 2)
            
            diameter_px = prop.equivalent_diameter
            diameter_um = diameter_px * self.pixel_to_um
            
            if diameter_um < min_diameter_um:
                continue
            
            perimeter_px = prop.perimeter
            circularity = 4 * np.pi * area_px / (perimeter_px ** 2) if perimeter_px > 0 else 0
            
            measurements.append({
                'organoid_id': int(prop.label),
                'area_pixels': int(area_px),
                'area_um2': round(area_um2, 2),
                'diameter_pixels': round(diameter_px, 2),
                'diameter_um': round(diameter_um, 2),
                'perimeter_pixels': round(perimeter_px, 2),
                'perimeter_um': round(perimeter_px * self.pixel_to_um, 2),
                'centroid': prop.centroid,
                'bbox': prop.bbox,
                'eccentricity': round(prop.eccentricity, 3),
                'solidity': round(prop.solidity, 3),
                'circularity': round(min(circularity, 1.0), 3)
            })
        
        return pd.DataFrame(measurements)
    
    def compare_conditions(self, control_df, experimental_df):
        """Statistical comparison between groups"""
        results = {}
        
        for name, df in [('control', control_df), ('experimental', experimental_df)]:
            if len(df) > 0:
                results[f'{name}_stats'] = {
                    'count': len(df),
                    'mean_diameter_um': round(df['diameter_um'].mean(), 2),
                    'std_diameter_um': round(df['diameter_um'].std(), 2),
                    'median_diameter_um': round(df['diameter_um'].median(), 2),
                    'min_diameter_um': round(df['diameter_um'].min(), 2),
                    'max_diameter_um': round(df['diameter_um'].max(), 2),
                    'mean_area_um2': round(df['area_um2'].mean(), 2),
                    'mean_circularity': round(df['circularity'].mean(), 3),
                }
            else:
                results[f'{name}_stats'] = {
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
    
    def create_overlay(self, image, masks, measurements_df, show_labels=True):
        """Create visualization with boundaries"""
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = image.copy()
        
        overlay = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if masks is None or len(measurements_df) == 0:
            cv2.putText(overlay, "No organoids detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return overlay
        
        # Generate colors
        n = len(measurements_df)
        colors = []
        for i in range(n):
            hue = int(180 * i / max(n, 1))
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr)))
        
        # Draw each organoid
        for idx, (_, row) in enumerate(measurements_df.iterrows()):
            org_id = int(row['organoid_id'])
            color = colors[idx]
            
            mask = (masks == org_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Fill with transparency
            mask_colored = np.zeros_like(overlay)
            cv2.drawContours(mask_colored, contours, -1, color, -1)
            overlay = cv2.addWeighted(overlay, 1, mask_colored, 0.35, 0)
            
            # Draw thick border
            cv2.drawContours(overlay, contours, -1, color, 3)
            
            # Label
            if show_labels and contours:
                cy, cx = int(row['centroid'][0]), int(row['centroid'][1])
                text = f"#{org_id}: {row['diameter_um']:.0f}um"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.4
                (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
                
                # Background
                cv2.rectangle(overlay, (cx - tw//2 - 2, cy - th - 4),
                             (cx + tw//2 + 2, cy + 4), (0, 0, 0), -1)
                # Text
                cv2.putText(overlay, text, (cx - tw//2, cy),
                           font, scale, (255, 255, 255), 1)
        
        # Count
        cv2.putText(overlay, f"Count: {n}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return overlay
    
    def create_boundary_overlay(self, image, masks, measurements_df):
        """Create clean boundary-only visualization"""
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = image.copy()
        
        overlay = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if masks is None or len(measurements_df) == 0:
            return overlay
        
        n = len(measurements_df)
        for idx, (_, row) in enumerate(measurements_df.iterrows()):
            org_id = int(row['organoid_id'])
            
            # Color
            hue = int(180 * idx / max(n, 1))
            hsv = np.uint8([[[hue, 255, 255]]])
            color = tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]))
            
            mask = (masks == org_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Thick contour
            cv2.drawContours(overlay, contours, -1, color, 3)
            
            # Number label
            cy, cx = int(row['centroid'][0]), int(row['centroid'][1])
            cv2.circle(overlay, (cx, cy), 12, (0, 0, 0), -1)
            cv2.circle(overlay, (cx, cy), 12, color, 2)
            
            text = str(org_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
            cv2.putText(overlay, text, (cx - tw//2, cy + th//2),
                       font, 0.4, (255, 255, 255), 1)
        
        cv2.putText(overlay, f"Count: {n}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return overlay


# Backward compatibility - keep the old class name working
MorphologyAnalyzer = OrganoidAnalyzer


# ============ TESTING ============

def test_image(image_path, method='classical', min_size=500):
    """Test segmentation on an image"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"Method: {method}")
    print(f"{'='*60}\n")
    
    analyzer = OrganoidAnalyzer(method=method)
    analyzer.params['min_size_pixels'] = min_size
    
    masks, original = analyzer.segment_organoids(image_path, min_size=min_size)
    measurements = analyzer.measure_organoids(masks)
    
    print(f"\n✓ Detected {len(measurements)} organoids")
    
    if len(measurements) > 0:
        print(f"\nMeasurements:")
        print(measurements[['organoid_id', 'diameter_um', 'area_um2', 'circularity']].to_string())
        
        # Save results
        base = image_path.rsplit('.', 1)[0]
        
        overlay = analyzer.create_overlay(original, masks, measurements)
        cv2.imwrite(f"{base}_detected.png", overlay)
        print(f"\n✓ Saved: {base}_detected.png")
        
        boundary = analyzer.create_boundary_overlay(original, masks, measurements)
        cv2.imwrite(f"{base}_boundary.png", boundary)
        print(f"✓ Saved: {base}_boundary.png")
    
    return measurements


def compare_methods(image_path):
    """Compare different segmentation methods"""
    print(f"\n{'='*60}")
    print(f"COMPARING METHODS: {image_path}")
    print(f"{'='*60}\n")
    
    methods = ['classical', 'threshold']
    if CELLPOSE_AVAILABLE:
        methods.append('cellpose')
    if STARDIST_AVAILABLE:
        methods.append('stardist')
    
    results = []
    
    for method in methods:
        print(f"\n--- Testing {method} ---")
        try:
            analyzer = OrganoidAnalyzer(method=method)
            masks, original = analyzer.segment_organoids(image_path)
            count = masks.max() if masks is not None else 0
            results.append({'method': method, 'count': count})
            print(f"  → Detected: {count} organoids")
            
            # Save overlay
            if count > 0:
                measurements = analyzer.measure_organoids(masks)
                overlay = analyzer.create_overlay(original, masks, measurements)
                base = image_path.rsplit('.', 1)[0]
                cv2.imwrite(f"{base}_{method}.png", overlay)
        except Exception as e:
            print(f"  → Error: {e}")
            results.append({'method': method, 'count': -1})
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS:")
    print(pd.DataFrame(results).to_string())
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python morphology_analyzer.py <image_path>")
        print("  python morphology_analyzer.py <image_path> --compare")
        print("  python morphology_analyzer.py <image_path> --method classical")
        print("  python morphology_analyzer.py <image_path> --method stardist")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if '--compare' in sys.argv:
        compare_methods(image_path)
    elif '--method' in sys.argv:
        idx = sys.argv.index('--method')
        method = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'classical'
        test_image(image_path, method=method)
    else:
        test_image(image_path, method='classical')