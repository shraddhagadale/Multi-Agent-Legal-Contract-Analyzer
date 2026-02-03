"""
PDF Report Generator for LegalDoc AI

Generates professional PDF reports from contract analysis results.
Uses ReportLab for PDF generation.
"""

from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


class PDFReportGenerator:
    """
    Generates PDF reports from contract analysis results.
    
    The report follows this structure:
    1. Header with document metadata
    2. Executive summary with risk distribution
    3. High risk clauses (full detail)
    4. Medium risk clauses (summary table)
    5. Low risk clauses (simple list)
    6. Footer
    """
    
    # Color scheme
    COLORS = {
        'high': colors.HexColor('#DC2626'),      # Red
        'medium': colors.HexColor('#D97706'),    # Orange/Yellow
        'low': colors.HexColor('#16A34A'),       # Green
        'header_bg': colors.HexColor('#1F2937'), # Dark gray
        'light_gray': colors.HexColor('#F3F4F6'),
        'border': colors.HexColor('#E5E7EB'),
    }
    
    def __init__(self, results: Dict[str, Any], metadata: Dict[str, Any]):
        """
        Initialize the PDF generator.
        
        Args:
            results: Analysis results from LegalDocAI
            metadata: Report metadata (source_file, analyzed_at, provider)
        """
        self.results = results
        self.metadata = metadata
        self.styles = self._create_styles()
        self.elements: List = []
    
    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for the report."""
        base_styles = getSampleStyleSheet()
        
        styles = {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=base_styles['Heading1'],
                fontSize=24,
                spaceAfter=6,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#111827'),
            ),
            'subtitle': ParagraphStyle(
                'CustomSubtitle',
                parent=base_styles['Normal'],
                fontSize=11,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#6B7280'),
                spaceAfter=20,
            ),
            'section_header': ParagraphStyle(
                'SectionHeader',
                parent=base_styles['Heading2'],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=12,
                textColor=colors.HexColor('#111827'),
            ),
            'clause_title': ParagraphStyle(
                'ClauseTitle',
                parent=base_styles['Heading3'],
                fontSize=13,
                spaceBefore=12,
                spaceAfter=6,
                textColor=colors.HexColor('#1F2937'),
            ),
            'body': ParagraphStyle(
                'CustomBody',
                parent=base_styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                leading=14,
            ),
            'clause_text': ParagraphStyle(
                'ClauseText',
                parent=base_styles['Normal'],
                fontSize=9,
                leftIndent=20,
                rightIndent=20,
                spaceAfter=10,
                backColor=colors.HexColor('#F9FAFB'),
                borderPadding=10,
                leading=13,
                textColor=colors.HexColor('#374151'),
            ),
            'risk_item': ParagraphStyle(
                'RiskItem',
                parent=base_styles['Normal'],
                fontSize=10,
                leftIndent=20,
                spaceAfter=4,
                leading=13,
            ),
            'recommendation': ParagraphStyle(
                'Recommendation',
                parent=base_styles['Normal'],
                fontSize=10,
                leftIndent=20,
                spaceAfter=3,
                bulletIndent=10,
                leading=13,
            ),
            'footer': ParagraphStyle(
                'Footer',
                parent=base_styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#9CA3AF'),
            ),
            'label': ParagraphStyle(
                'Label',
                parent=base_styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#6B7280'),
            ),
        }
        
        return styles
    
    def generate(self, output_path: str) -> str:
        """
        Generate the PDF report.
        
        Args:
            output_path: Path to save the PDF file
            
        Returns:
            Path to the generated PDF file
        """
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )
        
        # Build report sections
        self._add_header()
        self._add_summary()
        self._add_high_risk_section()
        self._add_medium_risk_section()
        self._add_low_risk_section()
        self._add_footer()
        
        # Generate PDF
        doc.build(self.elements)
        
        return output_path
    
    def _add_header(self) -> None:
        """Add report header with title and metadata."""
        # Title
        self.elements.append(Paragraph("NDA Risk Analysis Report", self.styles['title']))
        
        # Metadata
        source_file = self.metadata.get('source_file', 'Unknown')
        analyzed_at = self.metadata.get('analyzed_at', datetime.now().isoformat())
        provider = self.metadata.get('provider', 'Unknown')
        
        # Format date nicely
        try:
            dt = datetime.fromisoformat(analyzed_at.replace('Z', '+00:00'))
            formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
        except (ValueError, AttributeError):
            formatted_date = analyzed_at
        
        metadata_text = f"Document: {source_file} | Analyzed: {formatted_date} | Provider: {provider}"
        self.elements.append(Paragraph(metadata_text, self.styles['subtitle']))
        
        # Divider
        self.elements.append(HRFlowable(width="100%", color=self.COLORS['border']))
        self.elements.append(Spacer(1, 12))
    
    def _add_summary(self) -> None:
        """Add executive summary with risk distribution."""
        self.elements.append(Paragraph("Executive Summary", self.styles['section_header']))
        
        total_clauses = self.results.get('total_clauses', 0)
        high_count = len(self.results.get('high_risk_clauses', []))
        medium_count = len(self.results.get('medium_risk_clauses', []))
        low_count = len(self.results.get('low_risk_clauses', []))
        
        # Summary text
        summary_text = f"<b>Total Clauses Analyzed:</b> {total_clauses}"
        self.elements.append(Paragraph(summary_text, self.styles['body']))
        self.elements.append(Spacer(1, 8))
        
        # Risk distribution table
        risk_data = [
            ['Risk Level', 'Count', 'Status'],
            ['HIGH RISK', str(high_count), 'Action Required'],
            ['MEDIUM RISK', str(medium_count), 'Review Recommended'],
            ['LOW RISK', str(low_count), 'Acceptable'],
        ]
        
        table = Table(risk_data, colWidths=[2*inch, 1*inch, 2*inch])
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['header_bg']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
            # Data rows
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#FEE2E2')),  # High - light red
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FEF3C7')),  # Medium - light yellow
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#DCFCE7')),  # Low - light green
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['border']),
        ]))
        
        self.elements.append(table)
        self.elements.append(Spacer(1, 20))
    
    def _add_high_risk_section(self) -> None:
        """Add detailed high risk clauses section."""
        high_risk_clauses = self.results.get('high_risk_clauses', [])
        
        if not high_risk_clauses:
            return
        
        self.elements.append(Paragraph("High Risk Clauses (Action Required)", self.styles['section_header']))
        self.elements.append(HRFlowable(width="100%", color=self.COLORS['high'], thickness=2))
        self.elements.append(Spacer(1, 12))
        
        for i, clause in enumerate(high_risk_clauses, 1):
            self._add_high_risk_clause(i, clause)
    
    def _add_high_risk_clause(self, index: int, clause: Dict[str, Any]) -> None:
        """Add a single high risk clause with full details."""
        # Get clause info from the nested structure
        original_clause = clause.get('original_clause', {})
        clause_title = original_clause.get('clause_title', f"Clause {index}")
        clause_id = original_clause.get('clause_id', '')
        clause_text = original_clause.get('clause_text', '')
        
        risk_score = clause.get('risk_score', 0)
        assessment = clause.get('overall_assessment', '')
        risks = clause.get('identified_risks', [])
        recommendations = clause.get('recommendations', [])
        
        # Clause header
        header_text = f"{index}. {clause_title}"
        if clause_id:
            header_text += f" (ID: {clause_id})"
        self.elements.append(Paragraph(header_text, self.styles['clause_title']))
        
        # Risk score
        score_text = f"<b>Risk Score:</b> {risk_score:.2f}"
        self.elements.append(Paragraph(score_text, self.styles['body']))
        
        # Assessment
        if assessment:
            assessment_text = f"<b>Assessment:</b> {self._escape_html(assessment)}"
            self.elements.append(Paragraph(assessment_text, self.styles['body']))
        
        self.elements.append(Spacer(1, 6))
        
        # Original text
        self.elements.append(Paragraph("<b>Original Text:</b>", self.styles['label']))
        # Truncate long text
        display_text = clause_text[:500] + "..." if len(clause_text) > 500 else clause_text
        self.elements.append(Paragraph(self._escape_html(display_text), self.styles['clause_text']))
        
        # Identified risks
        if risks:
            self.elements.append(Paragraph("<b>Identified Risks:</b>", self.styles['label']))
            for risk in risks:
                risk_type = risk.get('risk_type', 'Unknown')
                severity = risk.get('severity', 'UNKNOWN')
                description = risk.get('description', '')
                
                severity_color = '#DC2626' if severity == 'HIGH' else '#D97706' if severity == 'MEDIUM' else '#16A34A'
                risk_text = f"* <font color='{severity_color}'><b>{risk_type} ({severity})</b></font><br/>{self._escape_html(description)}"
                self.elements.append(Paragraph(risk_text, self.styles['risk_item']))
            self.elements.append(Spacer(1, 6))
        
        # Recommendations
        if recommendations:
            self.elements.append(Paragraph("<b>Recommendations:</b>", self.styles['label']))
            for rec in recommendations:
                rec_text = f"- {self._escape_html(rec)}"
                self.elements.append(Paragraph(rec_text, self.styles['recommendation']))
        
        # Divider between clauses
        self.elements.append(Spacer(1, 10))
        self.elements.append(HRFlowable(width="100%", color=self.COLORS['border']))
        self.elements.append(Spacer(1, 10))
    
    def _add_medium_risk_section(self) -> None:
        """Add medium risk clauses as summary table."""
        medium_risk_clauses = self.results.get('medium_risk_clauses', [])
        
        if not medium_risk_clauses:
            return
        
        self.elements.append(Paragraph("Medium Risk Clauses (Review Recommended)", self.styles['section_header']))
        self.elements.append(HRFlowable(width="100%", color=self.COLORS['medium'], thickness=2))
        self.elements.append(Spacer(1, 12))
        
        # Build table data
        table_data = [['#', 'Clause', 'Risk Score', 'Primary Concern']]
        
        for i, clause in enumerate(medium_risk_clauses, 1):
            original_clause = clause.get('original_clause', {})
            clause_title = original_clause.get('clause_title', f"Clause {i}")
            risk_score = clause.get('risk_score', 0)
            
            # Get primary concern (first risk or assessment summary)
            risks = clause.get('identified_risks', [])
            if risks:
                primary_concern = risks[0].get('risk_type', 'Review needed')
            else:
                assessment = clause.get('overall_assessment', 'Review needed')
                primary_concern = assessment[:50] + "..." if len(assessment) > 50 else assessment
            
            table_data.append([str(i), clause_title, f"{risk_score:.2f}", primary_concern])
        
        # Create table
        table = Table(table_data, colWidths=[0.4*inch, 2*inch, 0.8*inch, 3.5*inch])
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['medium']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('ALIGN', (2, 1), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLORS['light_gray']]),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['border']),
        ]))
        
        self.elements.append(table)
        self.elements.append(Spacer(1, 20))
    
    def _add_low_risk_section(self) -> None:
        """Add low risk clauses as simple list."""
        low_risk_clauses = self.results.get('low_risk_clauses', [])
        
        if not low_risk_clauses:
            return
        
        self.elements.append(Paragraph("Low Risk Clauses (Acceptable)", self.styles['section_header']))
        self.elements.append(HRFlowable(width="100%", color=self.COLORS['low'], thickness=2))
        self.elements.append(Spacer(1, 12))
        
        # Simple list
        for clause in low_risk_clauses:
            original_clause = clause.get('original_clause', {})
            clause_title = original_clause.get('clause_title', 'Unknown Clause')
            clause_id = original_clause.get('clause_id', '')
            risk_score = clause.get('risk_score', 0)
            
            item_text = f"* {clause_title}"
            if clause_id:
                item_text += f" ({clause_id})"
            item_text += f" - Score: {risk_score:.2f}"
            
            self.elements.append(Paragraph(item_text, self.styles['body']))
        
        self.elements.append(Spacer(1, 20))
    
    def _add_footer(self) -> None:
        """Add report footer."""
        self.elements.append(Spacer(1, 20))
        self.elements.append(HRFlowable(width="100%", color=self.COLORS['border']))
        self.elements.append(Spacer(1, 10))
        
        footer_text = f"Generated by LegalDoc AI | {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        self.elements.append(Paragraph(footer_text, self.styles['footer']))
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters for ReportLab."""
        if not text:
            return ""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('\n', '<br/>'))
