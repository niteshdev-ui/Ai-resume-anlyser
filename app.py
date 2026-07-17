import io
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.parser import ParsedResume, ResumeParser
from modules.ai_analysis import ResumeAnalyzer
from modules.reporting import ReportGenerator


st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")


@st.cache_data(show_spinner=False)
def load_sample_dataset():
    sample_dir = Path(__file__).parent / "sample_data"
    if not sample_dir.exists():
        return [], ""

    resumes: List[ParsedResume] = []
    parser = ResumeParser()
    for sample_path in sorted(sample_dir.glob("resume*.txt")):
        text = sample_path.read_text(encoding="utf-8")
        resumes.append(parser.parse_text(sample_path.name, text))

    job_description = (sample_dir / "job_description.txt").read_text(encoding="utf-8")
    return resumes, job_description


def render_gauge(score: float, label: str):
    score_value = max(0, min(100, int(score)))
    st.markdown(
        f"""
        <div style="max-width:220px; margin:0 auto; text-align:center;">
            <div style="font-size:16px; font-weight:600; color:#0f172a;">{label}</div>
            <div style="margin-top:10px; height:140px; border-radius:50%; border:14px solid #e2e8f0; position:relative;">
                <div style="position:absolute; inset:0; border-radius:50%; background:conic-gradient(#2563eb 0 {score_value}%, #f1f5f9 {score_value}% 100%);"></div>
                <div style="position:absolute; inset:16px; border-radius:50%; background:white; display:flex; align-items:center; justify-content:center; font-size:28px; font-weight:700; color:#0f172a;">{score_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("🤖 AI Resume Analyzer")
    st.caption("Analyze resumes against a job description, score ATS readiness, and generate actionable recommendations.")

    with st.sidebar:
        st.header("Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        st.text_area(
            "Job Description",
            key="job_description",
            height=220,
            placeholder="Paste the target job description here...",
        )

        if st.button("Load Sample Dataset", use_container_width=True):
            sample_resumes, sample_job_description = load_sample_dataset()
            if sample_resumes:
                st.session_state["sample_resumes"] = sample_resumes
                st.session_state["job_description"] = sample_job_description
                st.success("Sample dataset loaded.")
            else:
                st.warning("Sample dataset not found.")

        analyze_clicked = st.button("Analyze Resumes", type="primary", use_container_width=True)

    if analyze_clicked or "analysis_results" in st.session_state:
        parser = ResumeParser()

        resumes: List[ParsedResume] = []
        if "sample_resumes" in st.session_state and st.session_state["sample_resumes"]:
            resumes = st.session_state["sample_resumes"]
        elif uploaded_files:
            for uploaded_file in uploaded_files:
                parsed_resume = parser.parse(uploaded_file)
                resumes.append(parsed_resume)

        if not resumes:
            st.info("Upload one or more resumes or load the sample dataset to begin.")
            return

        job_description = st.session_state.get("job_description", "").strip()
        if not job_description:
            st.warning("Please provide a job description before analyzing.")
            return

        with st.spinner("Analyzing resumes and generating insights..."):
            analyzer = ResumeAnalyzer()
            analysis_results = analyzer.analyze(resumes, job_description)
            st.session_state["analysis_results"] = analysis_results

        results = st.session_state["analysis_results"]

        st.header("📊 Dashboard")
        top_result = results[0]
        cols = st.columns(4)
        cols[0].metric("Top ATS Score", f"{top_result.ats_score:.1f}/100")
        cols[1].metric("Skill Match", f"{top_result.skill_match_percentage:.1f}%")
        cols[2].metric("Resume Strength", f"{top_result.resume_strength_score:.1f}/100")
        cols[3].metric("Matched Skills", len(top_result.matched_skills))

        col_a, col_b = st.columns([1, 1])
        with col_a:
            render_gauge(top_result.ats_score, "ATS Score")
        with col_b:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(
                [top_result.skill_match_percentage, 100 - top_result.skill_match_percentage],
                labels=["Matched", "Gap"],
                autopct="%1.1f%%",
                colors=["#2563eb", "#dbeafe"],
                startangle=90,
            )
            ax.set_title("Skill Match Distribution")
            st.pyplot(fig)

        st.subheader("Resume Ranking")
        ranking_data = []
        for item in results:
            ranking_data.append(
                {
                    "Candidate": item.candidate_name,
                    "ATS Score": round(item.ats_score, 1),
                    "Skill Match": round(item.skill_match_percentage, 1),
                    "Strength": round(item.resume_strength_score, 1),
                    "Missing Skills": len(item.missing_skills),
                }
            )

        ranking_df = pd.DataFrame(ranking_data).sort_values("ATS Score", ascending=False)
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

        st.subheader("Skill Analysis")
        skill_col, missing_col = st.columns(2)
        with skill_col:
            st.markdown("**Matched Skills**")
            for skill in top_result.matched_skills[:10]:
                st.write(f"- {skill}")
        with missing_col:
            st.markdown("**Missing Skills**")
            if top_result.missing_skills:
                missing_labels = [skill for skill in top_result.missing_skills[:10]]
                missing_values = [1] * len(missing_labels)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(missing_labels[::-1], missing_values[::-1], color="#f59e0b")
                ax.set_title("Missing Skills")
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.write("No missing skills detected.")

        st.subheader("Recommendations")
        recs = top_result.recommendations
        if recs:
            for item in recs:
                st.write(f"- {item}")
        else:
            st.write("No recommendations generated.")

        st.subheader("AI Summary")
        st.write(top_result.summary)

        st.subheader("Resume Statistics")
        stats_cols = st.columns(4)
        stats_cols[0].metric("Name", top_result.candidate_name or "Unknown")
        stats_cols[1].metric("Email", top_result.email or "Not detected")
        stats_cols[2].metric("Phone", top_result.phone_number or "Not detected")
        stats_cols[3].metric("Section Quality", f"{top_result.section_quality_score:.1f}/100")

        st.subheader("Additional Insights")
        insight_cols = st.columns(3)
        insight_cols[0].metric("Keyword Density", f"{top_result.keyword_density:.1f}%")
        insight_cols[1].metric("Grammar Score", f"{top_result.grammar_score:.1f}/100")
        insight_cols[2].metric("Readability Score", f"{top_result.readability_score:.1f}/100")

        report_generator = ReportGenerator()
        report_bytes = report_generator.generate_pdf(top_result, job_description)
        st.download_button(
            label="Download PDF Report",
            data=report_bytes,
            file_name=f"{top_result.candidate_name.replace(' ', '_')}_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
