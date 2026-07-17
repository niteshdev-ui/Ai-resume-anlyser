# AI Resume Analyzer

A production-ready Python and Streamlit application for analyzing resumes against a target job description. The app supports PDF, DOCX, and TXT uploads, extracts profile fields, computes ATS and skill match scores, generates recommendations, and produces downloadable PDF reports.

## Features

- Modern Streamlit UI
- Resume upload for PDF, DOCX, and TXT
- Job description analysis
- Name, email, phone, skills, education, and experience extraction
- ATS scoring, skill match, and resume strength scoring
- Recommendation and career guidance generation
- Downloadable PDF reports
- Sample dataset support

## Project Structure

```text
ai_resume_analyzer/
├── app.py
├── requirements.txt
├── README.md
├── sample_data/
│   ├── resume_1.txt
│   ├── resume_2.txt
│   ├── job_description.txt
├── modules/
│   ├── __init__.py
│   ├── parser.py
│   ├── ai_analysis.py
│   └── reporting.py
```

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

- Upload one or more resumes.
- Paste a job description.
- Click Analyze Resumes.
- Review the dashboard, recommendations, and download the PDF report.
