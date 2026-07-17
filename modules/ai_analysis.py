import re
from typing import List, Optional

import spacy
from sentence_transformers import SentenceTransformer

from modules.parser import ParsedResume


class ResumeAnalyzer:
    """Analyze resume content against a job description using NLP and semantic similarity."""

    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.sentence_model: Optional[SentenceTransformer] = None

    def _load_spacy_model(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return spacy.blank("en")

    def _load_sentence_model(self):
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.sentence_model

    def analyze(self, resumes: List[ParsedResume], job_description: str) -> List[ParsedResume]:
        job_skills = self._extract_job_skills(job_description)
        results = []
        for resume in resumes:
            result = self._analyze_single_resume(resume, job_description, job_skills)
            results.append(result)
        return sorted(results, key=lambda item: item.ats_score, reverse=True)

    def _analyze_single_resume(self, resume: ParsedResume, job_description: str, job_skills: List[str]):
        resume_text = resume.text.lower()
        job_text = job_description.lower()
        semantic_similarity = self._compute_semantic_similarity(resume.text, job_description)
        matched_skills = [skill for skill in job_skills if skill.lower() in resume_text]
        missing_skills = [skill for skill in job_skills if skill.lower() not in resume_text]
        keyword_density = self._compute_keyword_density(resume.text, job_description)
        section_quality_score = self._compute_section_quality(resume)
        grammar_score = self._compute_grammar_score(resume.text)
        readability_score = self._compute_readability_score(resume.text)
        skill_match_percentage = (len(matched_skills) / max(len(job_skills), 1)) * 100 if job_skills else 100.0
        ats_score = self._compute_ats_score(
            semantic_similarity,
            skill_match_percentage,
            section_quality_score,
            keyword_density,
            readability_score,
        )
        resume_strength_score = self._compute_resume_strength_score(resume, matched_skills, section_quality_score)
        recommendations = self._generate_recommendations(missing_skills, resume, job_description)
        career_recommendations = self._generate_career_recommendations(resume, missing_skills)
        summary = self._generate_summary(resume, matched_skills, missing_skills, ats_score)

        resume.ats_score = ats_score
        resume.skill_match_percentage = skill_match_percentage
        resume.resume_strength_score = resume_strength_score
        resume.matched_skills = matched_skills
        resume.missing_skills = missing_skills
        resume.keywords_to_add = missing_skills[:5]
        resume.missing_sections = self._detect_missing_sections(resume)
        resume.recommendations = recommendations
        resume.career_recommendations = career_recommendations
        resume.summary = summary
        resume.keyword_density = keyword_density
        resume.section_quality_score = section_quality_score
        resume.grammar_score = grammar_score
        resume.readability_score = readability_score
        return resume

    def _extract_job_skills(self, job_description: str) -> List[str]:
        keywords = [
            "python",
            "java",
            "javascript",
            "typescript",
            "sql",
            "aws",
            "azure",
            "docker",
            "kubernetes",
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "streamlit",
            "flask",
            "django",
            "fastapi",
            "pandas",
            "numpy",
            "tableau",
            "power bi",
            "spark",
            "nlp",
            "spacy",
            "git",
            "docker",
            "api",
            "react",
            "node.js",
        ]
        found = []
        lower_text = job_description.lower()
        for keyword in keywords:
            if keyword.lower() in lower_text:
                found.append(keyword)
        return found

    def _compute_semantic_similarity(self, resume_text: str, job_description: str) -> float:
        try:
            model = self._load_sentence_model()
            embeddings = model.encode([resume_text, job_description])
            dot = float(embeddings[0] @ embeddings[1])
            norm = (embeddings[0] @ embeddings[0]) ** 0.5 * (embeddings[1] @ embeddings[1]) ** 0.5
            if norm == 0:
                return 0.0
            return max(0.0, min(1.0, dot / norm))
        except Exception:
            tokens_resume = set(re.findall(r"[a-zA-Z]{3,}", resume_text.lower()))
            tokens_job = set(re.findall(r"[a-zA-Z]{3,}", job_description.lower()))
            if not tokens_resume or not tokens_job:
                return 0.0
            overlap = len(tokens_resume & tokens_job) / max(len(tokens_job), 1)
            return max(0.0, min(1.0, overlap))

    def _compute_keyword_density(self, resume_text: str, job_description: str) -> float:
        job_keywords = set(re.findall(r"[a-zA-Z]{3,}", job_description.lower()))
        resume_tokens = re.findall(r"[a-zA-Z]{3,}", resume_text.lower())
        if not job_keywords:
            return 100.0
        matches = sum(1 for token in resume_tokens if token in job_keywords)
        return round((matches / max(len(job_keywords), 1)) * 100, 1)

    def _compute_section_quality(self, resume: ParsedResume) -> float:
        score = 0.0
        if resume.name:
            score += 20
        if resume.email:
            score += 15
        if resume.phone_number:
            score += 15
        if resume.skills:
            score += 20
        if resume.education:
            score += 15
        if resume.experience:
            score += 15
        return min(100.0, score)

    def _compute_grammar_score(self, text: str) -> float:
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return 60.0
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        score = max(0.0, 100 - abs(avg_len - 18) * 3)
        return round(min(100.0, score), 1)

    def _compute_readability_score(self, text: str) -> float:
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return 70.0
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        score = max(0.0, 100 - abs(avg_len - 16) * 2.5)
        return round(min(100.0, score), 1)

    def _compute_ats_score(self, semantic_similarity, skill_match_percentage, section_quality_score, keyword_density, readability_score) -> float:
        score = (
            semantic_similarity * 0.35
            + (skill_match_percentage / 100) * 0.30
            + (section_quality_score / 100) * 0.15
            + (keyword_density / 100) * 0.10
            + (readability_score / 100) * 0.10
        ) * 100
        return round(max(0.0, min(100.0, score)), 1)

    def _compute_resume_strength_score(self, resume: ParsedResume, matched_skills: List[str], section_quality_score: float) -> float:
        strength = section_quality_score * 0.6
        strength += min(100.0, len(matched_skills) * 10) * 0.4
        return round(max(0.0, min(100.0, strength)), 1)

    def _detect_missing_sections(self, resume: ParsedResume) -> List[str]:
        missing = []
        if not resume.skills:
            missing.append("Skills")
        if not resume.education:
            missing.append("Education")
        if not resume.experience:
            missing.append("Experience")
        if not resume.summary:
            missing.append("Professional Summary")
        return missing

    def _generate_recommendations(self, missing_skills: List[str], resume: ParsedResume, job_description: str) -> List[str]:
        recommendations = []
        if missing_skills:
            recommendations.append(f"Add these keywords to the resume: {', '.join(missing_skills[:5])}.")
        if not resume.skills:
            recommendations.append("Create a dedicated skills section with your technical stack and tools.")
        if not resume.education:
            recommendations.append("Include academic credentials and relevant coursework or certifications.")
        if not resume.experience:
            recommendations.append("Expand the experience section with measurable achievements and project outcomes.")
        if len(resume.text.split()) < 250:
            recommendations.append("Enrich the resume with more detail and quantifiable accomplishments.")
        return recommendations

    def _generate_career_recommendations(self, resume: ParsedResume, missing_skills: List[str]) -> List[str]:
        suggestions = []
        if "python" in [skill.lower() for skill in resume.skills] or "machine learning" in [skill.lower() for skill in resume.skills]:
            suggestions.append("Consider roles such as Python Developer, Data Scientist, or ML Engineer.")
        if any(skill.lower() in {"aws", "azure", "docker", "kubernetes"} for skill in resume.skills):
            suggestions.append("Consider cloud or DevOps-oriented opportunities.")
        if missing_skills:
            suggestions.append("Target roles that emphasize the missing technical skills in your profile.")
        return suggestions or ["Continue building your skill profile and target adjacent roles in the same domain."]

    def _generate_summary(self, resume: ParsedResume, matched_skills: List[str], missing_skills: List[str], ats_score: float) -> str:
        summary = [
            f"{resume.name or resume.filename} shows a strong ATS readiness score of {ats_score:.1f}/100.",
            f"The profile matches {len(matched_skills)} requested skills and has {len(missing_skills)} opportunities for improvement.",
        ]
        return " ".join(summary)
