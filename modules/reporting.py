import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class ReportGenerator:
    """Generate a downloadable PDF report for a single resume analysis."""

    def generate_pdf(self, analysis_result: Any, job_description: str) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("AI Resume Analyzer Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Candidate: {analysis_result.name or analysis_result.filename}", styles["Heading2"]))
        story.append(Paragraph(f"ATS Score: {analysis_result.ats_score:.1f}/100", styles["Heading3"]))
        story.append(Paragraph(f"Skill Match: {analysis_result.skill_match_percentage:.1f}%", styles["BodyText"]))
        story.append(Paragraph(f"Resume Strength: {analysis_result.resume_strength_score:.1f}/100", styles["BodyText"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Matched Skills", styles["Heading3"]))
        story.append(Paragraph(", ".join(analysis_result.matched_skills[:10]) or "No matched skills", styles["BodyText"]))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Missing Skills", styles["Heading3"]))
        story.append(Paragraph(", ".join(analysis_result.missing_skills[:10]) or "No missing skills", styles["BodyText"]))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Recommendations", styles["Heading3"]))
        for item in analysis_result.recommendations[:5]:
            story.append(Paragraph(f"- {item}", styles["BodyText"]))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Job Description", styles["Heading3"]))
        story.append(Paragraph(job_description[:900], styles["BodyText"]))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
