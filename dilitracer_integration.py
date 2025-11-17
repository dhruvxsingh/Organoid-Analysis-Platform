import torch
import tempfile
import os
from pathlib import Path
from dilipredict.models import DILIPredict
from dilipredict.image_loader import ImageLoader
from dilipredict.pipelines import DILIPredict as DILIPipeline

class DILITracerAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DILIPredict()
        self.img_loader = ImageLoader()
        self.pipeline = DILIPipeline(self.img_loader, self.model, self.device)
        self.temp_dir = tempfile.mkdtemp()
    
    def save_uploaded_files(self, uploaded_files):
        """Save uploaded files to temp directory and return paths"""
        file_paths = {}
        
        for day, files in uploaded_files.items():
            day_paths = []
            for file in files:
                # Save file to temp directory
                temp_path = Path(self.temp_dir) / day / file.name
                temp_path.parent.mkdir(exist_ok=True)
                
                with open(temp_path, 'wb') as f:
                    f.write(file.getbuffer())
                
                day_paths.append(str(temp_path))
            
            file_paths[day] = day_paths
        
        return file_paths
    
    def run_analysis(self, uploaded_files):
        """Run DILITracer analysis on uploaded files"""
        # Save files temporarily
        file_paths = self.save_uploaded_files(uploaded_files)
        
        # Prepare for model
        img_files = [
            file_paths.get('D00', []),
            file_paths.get('D01', []),
            file_paths.get('D02', []),
            file_paths.get('D03', [])
        ]
        
        # Run inference
        label, probabilities = self.pipeline(img_files)
        
        # Convert to readable format
        label_map = {0: "No-DILI", 1: "Less-DILI", 2: "Most-DILI"}
        
        return {
            'prediction': label_map[label],
            'label_numeric': label,
            'probabilities': {
                'No-DILI': float(probabilities[0]),
                'Less-DILI': float(probabilities[1]),
                'Most-DILI': float(probabilities[2])
            },
            'confidence': float(max(probabilities)),
            'risk_level': self._get_risk_level(label)
        }
    
    def _get_risk_level(self, label):
        risk_map = {
            0: "Low Risk",
            1: "Moderate Risk",
            2: "High Risk"
        }
        return risk_map.get(label, "Unknown")