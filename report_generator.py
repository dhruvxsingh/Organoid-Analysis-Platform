from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import tempfile

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1E3A8A'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=12
        )
        
    def generate_report(self, results, uploaded_files):
        # Create temporary file for PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Add title
        elements.append(Paragraph("DILITracer Analysis Report", self.title_style))
        elements.append(Spacer(1, 12))
        
        # Add date and basic info
        date_style = ParagraphStyle('DateStyle', parent=self.styles['Normal'], alignment=TA_CENTER)
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", date_style))
        elements.append(Spacer(1, 20))
        
        # Add summary section
        elements.append(Paragraph("Executive Summary", self.heading_style))
        
        # Create summary table
        summary_data = [
            ['Prediction', results['prediction']],
            ['Risk Level', results['risk_level']],
            ['Confidence', f"{abs(results['confidence']*100):.1f}%"],
            ['Classification', self._get_classification_text(results['prediction'])]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 30))
        
        # Add probability scores section
        elements.append(Paragraph("Probability Scores", self.heading_style))
        
        prob_data = [
            ['Classification', 'Probability Score'],
            ['No-DILI', f"{results['probabilities']['No-DILI']:.4f}"],
            ['Less-DILI', f"{results['probabilities']['Less-DILI']:.4f}"],
            ['Most-DILI', f"{results['probabilities']['Most-DILI']:.4f}"]
        ]
        
        # Highlight the predicted row
        predicted_idx = {'No-DILI': 1, 'Less-DILI': 2, 'Most-DILI': 3}[results['prediction']]
        
        prob_table = Table(prob_data, colWidths=[3*inch, 3*inch])
        prob_table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, predicted_idx), (-1, predicted_idx), colors.HexColor('#F59E0B')),
        ]
        prob_table.setStyle(TableStyle(prob_table_style))
        
        elements.append(prob_table)
        elements.append(Spacer(1, 30))
        
        # Add interpretation
        elements.append(Paragraph("Clinical Interpretation", self.heading_style))
        
        interpretation = self._get_interpretation(results['prediction'])
        elements.append(Paragraph(interpretation['text'], self.styles['Normal']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Recommendation:</b> {interpretation['recommendation']}", self.styles['Normal']))
        elements.append(Spacer(1, 30))
        
        # Add data summary
        elements.append(Paragraph("Data Summary", self.heading_style))
        
        # Count uploaded files
        file_counts = []
        total_images = 0
        for day, files in uploaded_files.items():
            count = len(files) if files else 0
            total_images += count
            file_counts.append([day, str(count)])
        
        file_counts.append(['Total', str(total_images)])
        
        data_table = Table([['Time Point', 'Number of Images']] + file_counts, colWidths=[2*inch, 2*inch])
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(data_table)
        elements.append(Spacer(1, 30))
        
        # Add disclaimer
        elements.append(Paragraph("Disclaimer", self.heading_style))
        disclaimer_text = """This report is generated by an AI model for research purposes only. 
        The predictions should not be used as the sole basis for clinical decisions. 
        Please consult with qualified professionals for comprehensive toxicity assessment."""
        elements.append(Paragraph(disclaimer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        
        return temp_file.name
    
    def _get_classification_text(self, prediction):
        classifications = {
            "No-DILI": "No Drug-Induced Liver Injury Concern",
            "Less-DILI": "Less Drug-Induced Liver Injury Concern",
            "Most-DILI": "Most Drug-Induced Liver Injury Concern"
        }
        return classifications.get(prediction, "Unknown")
    
    def _get_interpretation(self, prediction):
        interpretations = {
            "No-DILI": {
                "text": "The analyzed compound demonstrates no significant hepatotoxic effects based on organoid morphology analysis. The organoids maintained healthy structure and viability throughout the 4-day treatment period.",
                "recommendation": "The compound appears safe for further development and testing. Continue with standard protocols."
            },
            "Less-DILI": {
                "text": "The analyzed compound shows moderate hepatotoxic effects. Some morphological changes were observed in the organoids, but they remained largely viable throughout the treatment period.",
                "recommendation": "Consider dose optimization studies. Monitor liver function markers closely in subsequent studies."
            },
            "Most-DILI": {
                "text": "The analyzed compound exhibits severe hepatotoxic effects. Significant organoid damage, morphological deterioration, and cell death were observed during the treatment period.",
                "recommendation": "High hepatotoxicity risk identified. Consider structural modification or alternative compounds."
            }
        }
        return interpretations.get(prediction, {"text": "Unknown classification", "recommendation": "Further analysis required"})