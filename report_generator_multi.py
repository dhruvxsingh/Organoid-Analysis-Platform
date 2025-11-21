from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import tempfile

class MultiDrugReportGenerator:
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
        
    def generate_comparison_report(self, results, uploaded_files):
        # Create temporary file for PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
        
        elements = []
        
        # Title
        elements.append(Paragraph("Multi-Drug DILITracer Comparison Report", self.title_style))
        elements.append(Spacer(1, 12))
        
        # Date and summary
        date_style = ParagraphStyle('DateStyle', parent=self.styles['Normal'], alignment=TA_CENTER)
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", date_style))
        elements.append(Paragraph(f"Conditions Analyzed: {', '.join(results.keys())}", date_style))
        elements.append(Spacer(1, 20))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", self.heading_style))
        
        # Summary table
        summary_data = [['Condition', 'Prediction', 'Risk Level', 'Confidence', 'Recommendation']]
        
        for condition, result in results.items():
            recommendation = "✅ SAFE" if result['prediction'] == "No-DILI" else \
                           "⚠️ CAUTION" if result['prediction'] == "Less-DILI" else "❌ ELIMINATE"
            
            summary_data.append([
                condition,
                result['prediction'],
                result['risk_level'],
                f"{result['confidence']*100:.1f}%",
                recommendation
            ])
        
        summary_table = Table(summary_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.2*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 30))
        
        # Detailed Results
        elements.append(Paragraph("Detailed Probability Analysis", self.heading_style))
        
        prob_data = [['Condition', 'No-DILI (%)', 'Less-DILI (%)', 'Most-DILI (%)']]
        
        for condition, result in results.items():
            prob_data.append([
                condition,
                f"{result['probabilities']['No-DILI']*100:.2f}",
                f"{result['probabilities']['Less-DILI']*100:.2f}",
                f"{result['probabilities']['Most-DILI']*100:.2f}"
            ])
        
        prob_table = Table(prob_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(prob_table)
        elements.append(Spacer(1, 30))
        
        # Recommendation
        elements.append(Paragraph("Drug Selection Recommendation", self.heading_style))
        
        # Sort by safety
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['probabilities']['No-DILI'], 
                              reverse=True)
        
        rec_text = "Based on the DILITracer analysis, the following ranking is recommended:\n\n"
        for rank, (condition, result) in enumerate(sorted_results, 1):
            rec_text += f"{rank}. {condition}: {result['prediction']} "
            rec_text += f"(No-DILI: {result['probabilities']['No-DILI']*100:.1f}%)\n"
        
        elements.append(Paragraph(rec_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        
        return temp_file.name