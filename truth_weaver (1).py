import os
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import warnings

# Colab helpers
from google.colab import files
from getpass import getpass

# Optional Gemini client
# pip install google-generative-ai  # run separately if needed
import google.generativeai as genai

warnings.filterwarnings("ignore")

# -------------------------
# Data Classes
# -------------------------
@dataclass
class RevealedTruth:
    programming_experience: str
    programming_language: str
    skill_mastery: str
    leadership_claims: str
    team_experience: str
    skills_and_keywords: List[str]

@dataclass
class DeceptionPattern:
    lie_type: str
    contradictory_claims: List[str]

@dataclass
class ShadowAnalysis:
    shadow_id: str
    revealed_truth: Dict
    deception_patterns: List[Dict]

# -------------------------
# TruthWeaver core
# -------------------------
class TruthWeaver:
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", None)
        self.use_gemini = bool(self.api_key)
        if self.use_gemini:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
        self.deception_indicators = {
            "high_tremor": 0.3,
            "excessive_pauses": 0.2,
            "fillers": 0.15,
            "pitch_variation": 0.1,
            "intensity_variation": 0.1,
            "word_repetition": 0.15,
        }

    def preprocess_transcript(self, text: str) -> Dict[str, Any]:
        pause_pattern = r"/pause_(\d+\.?\d*)s/"
        pauses = re.findall(pause_pattern, text)
        pause_count = len(pauses)
        avg_pause_duration = float(np.mean([float(p) for p in pauses])) if pauses else 0.0
        tremor_count = text.count("/tremor/")
        filler_pattern = r"/filler_[\w-]+/"
        filler_count = len(re.findall(filler_pattern, text))
        repeat_pattern = r"/repeat_word:[\w-]+/"
        repeat_count = len(re.findall(repeat_pattern, text))

        cleaned_text = text
        cleaned_text = re.sub(pause_pattern, "", cleaned_text)
        cleaned_text = re.sub(r"/tremor/", "", cleaned_text)
        cleaned_text = re.sub(filler_pattern, "", cleaned_text)
        cleaned_text = re.sub(repeat_pattern, "", cleaned_text)
        cleaned_text = re.sub(r"/cutoff/", "...", cleaned_text)
        cleaned_text = re.sub(r"<unk>", "[unclear]", cleaned_text)
        cleaned_text = re.sub(r"\[\d+\.\d+\]", "", cleaned_text)
        cleaned_text = " ".join(cleaned_text.split())

        return {
            "cleaned_text": cleaned_text,
            "pause_count": pause_count,
            "avg_pause_duration": avg_pause_duration,
            "tremor_count": tremor_count,
            "filler_count": filler_count,
            "repeat_count": repeat_count,
        }

    def calculate_deception_score(self, session_data: Dict[str, Any]) -> float:
        def get_val(d, raw_key, proc_key, default=0.0):
            if isinstance(d, pd.Series) or isinstance(d, dict):
                if raw_key in d and pd.notna(d[raw_key]):
                    return d[raw_key]
                if proc_key in d and pd.notna(d[proc_key]):
                    return d[proc_key]
            return default

        trem = float(get_val(session_data, "tremor", "tremor_count", 0.0))
        pause = float(get_val(session_data, "pause", "pause_count", 0.0))
        filler = float(get_val(session_data, "filler", "filler_count", 0.0))
        repeat_word = float(get_val(session_data, "repeat_word", "repeat_count", 0.0))
        pitch = float(get_val(session_data, "pitch", "pitch", np.nan))
        intensity = float(get_val(session_data, "intensity", "intensity", np.nan))

        score = 0.0
        if trem > 5:
            score += self.deception_indicators["high_tremor"]
        if pause > 2:
            score += self.deception_indicators["excessive_pauses"]
        if filler > 0:
            score += self.deception_indicators["fillers"]
        if not np.isnan(pitch) and pitch > 120:
            score += self.deception_indicators["pitch_variation"]
        if not np.isnan(intensity) and intensity > 0.1:
            score += self.deception_indicators["intensity_variation"]
        if repeat_word > 0:
            score += self.deception_indicators["word_repetition"]
        return min(max(score, 0.0), 1.0)

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        first_brace = text.find("{")
        if first_brace != -1:
            candidate = text[first_brace:]
            for end in range(len(candidate), 0, -1):
                try:
                    return json.loads(candidate[:end])
                except Exception:
                    continue
        return {}

    def analyze_with_gemini(self, sessions_text: List[Dict], shadow_id: str) -> Dict:
        if not self.use_gemini:
            return self._fallback_analysis(sessions_text)

        prompt = (
            f"Analyze shadow {shadow_id}. Output ONLY JSON with keys: programming_experience, programming_language, "
            "skill_mastery, leadership_claims, team_experience, skills_and_keywords (list), contradictions (list of {topic, claims}), deception_types (list)."
        )
        for i, s in enumerate(sessions_text, 1):
            prompt += f"\nSession {i} (score={s.get('deception_score', 0):.2f}): {s.get('cleaned_text','')}"

        try:
            response = self.model.generate_content(prompt)
            text_out = getattr(response, "text", None) or str(response)
            return self._extract_json_from_text(text_out)
        except Exception:
            return self._fallback_analysis(sessions_text)

    def _fallback_analysis(self, sessions_text: List[Dict]) -> Dict:
        all_texts = " ".join([s.get("cleaned_text", "") for s in sessions_text]).lower()
        years = re.findall(r"(\d+)\s*(?:years?|yrs?)", all_texts)
        experience = f"{years[0]} years" if years else "unclear"
        language = "python" if "python" in all_texts else "not specified"
        leadership = "genuine" if "led a team" in all_texts else ("fabricated" if "worked alone" in all_texts else "unclear")
        team_exp = "team" if "team" in all_texts else "unclear"
        contradictions = []
        if "worked alone" in all_texts and "led a team" in all_texts:
            contradictions.append({"topic": "team experience", "claims": ["worked alone", "led a team"]})
        return {
            "programming_experience": experience,
            "programming_language": language,
            "skill_mastery": "unclear",
            "leadership_claims": leadership,
            "team_experience": team_exp,
            "skills_and_keywords": [],
            "contradictions": contradictions,
            "deception_types": ["rule-based"],
        }

    def detect_contradictions(self, analysis_result: Dict) -> List[DeceptionPattern]:
        patterns = []
        for c in analysis_result.get("contradictions", []):
            patterns.append(DeceptionPattern(lie_type=f"{c.get('topic','unknown')}_contradiction", contradictory_claims=c.get("claims", [])))
        for t in analysis_result.get("deception_types", []):
            patterns.append(DeceptionPattern(lie_type=t, contradictory_claims=[]))
        return patterns

    def analyze_shadow_agent(self, df: pd.DataFrame, shadow_id: str) -> ShadowAnalysis:
        shadow_data = df[df["shadow_id"] == shadow_id].copy().sort_values("session_id")
        if shadow_data.empty:
            raise ValueError(f"No data for shadow {shadow_id}")

        sessions_analysis = []
        for _, row in shadow_data.iterrows():
            processed = self.preprocess_transcript(str(row.get("text", "")))
            combined = {**processed}
            for col in ["tremor", "pause", "filler", "repeat_word", "pitch", "intensity"]:
                if col in row and pd.notna(row[col]):
                    combined[col] = row[col]
            deception_score = self.calculate_deception_score(combined)
            processed["deception_score"] = deception_score
            processed["session_id"] = row.get("session_id")
            sessions_analysis.append(processed)

        gemini_analysis = self.analyze_with_gemini(sessions_analysis, shadow_id)
        revealed_truth = RevealedTruth(
            programming_experience=gemini_analysis.get("programming_experience", "unclear"),
            programming_language=gemini_analysis.get("programming_language", "not specified"),
            skill_mastery=gemini_analysis.get("skill_mastery", "unclear"),
            leadership_claims=gemini_analysis.get("leadership_claims", "unclear"),
            team_experience=gemini_analysis.get("team_experience", "unclear"),
            skills_and_keywords=gemini_analysis.get("skills_and_keywords", []),
        )
        deception_patterns = self.detect_contradictions(gemini_analysis)
        return ShadowAnalysis(shadow_id, asdict(revealed_truth), [asdict(p) for p in deception_patterns])

    def process_all_shadows(self, df: pd.DataFrame) -> List[Dict]:
        results = []
        for sid in df["shadow_id"].unique():
            try:
                analysis = self.analyze_shadow_agent(df, sid)
                results.append(asdict(analysis))
            except Exception as e:
                print(f"Error processing {sid}: {e}")
        return results

# -------------------------
# Mystic Transformer
# -------------------------
def normalize_text_field(x, default="unclear"):
    if x is None:
        return default
    if isinstance(x, list):
        return ", ".join(map(str, x)) if x else default
    return str(x)

def transform_entry(entry):
    sid = entry.get("shadow_id", "unknown")
    revealed = entry.get("revealed_truth", {})
    prog_exp = normalize_text_field(revealed.get("programming_experience"))
    prog_lang = normalize_text_field(revealed.get("programming_language", "not specified"))
    skill_mastery = normalize_text_field(revealed.get("skill_mastery"))
    leadership_claims = normalize_text_field(revealed.get("leadership_claims"))
    team_experience = normalize_text_field(revealed.get("team_experience"))
    skills_list = revealed.get("skills_and_keywords", [])

    raw_patterns = entry.get("deception_patterns", [])
    patterns = []
    for p in raw_patterns:
        lie = p.get("lie_type", "unknown")
        claims = p.get("contradictory_claims", [])
        if isinstance(claims, str):
            claims = [c.strip() for c in claims.split(",") if c.strip()]
        patterns.append({"lie_type": lie, "contradictory_claims": claims})

    return {
        "shadow_id": sid,
        "revealed_truth": {
            "programming_experience": prog_exp,
            "programming_language": prog_lang,
            "skill_mastery": skill_mastery,
            "leadership_claims": leadership_claims,
            "team_experience": team_experience,
            "skills and other keywords": skills_list,
        },
        "deception_patterns": patterns,
    }

# -------------------------
# Colab Flow
# -------------------------
def run_colab_flow():
    print("Upload your CSV file...")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No file uploaded")

    csv_name = list(uploaded.keys())[0]
    df = pd.read_csv(csv_name)
    req_cols = ["shadow_id", "session_id", "text"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c}")
    for col in ["tremor", "pause", "filler", "repeat_word", "pitch", "intensity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip() or getpass("Enter Gemini API key (press Enter to skip): ").strip()
    tw = TruthWeaver(api_key=api_key)
    results = tw.process_all_shadows(df)

    # Save TruthWeaver raw results
    with open("truth_weaver_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Transform into mystical JSON
    mystic_list = [transform_entry(it) for it in results]
    with open("truth_weaver_mystic.json", "w", encoding="utf-8") as f:
        json.dump(mystic_list, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(mystic_list)} entries to truth_weaver_mystic.json")
    files.download("truth_weaver_mystic.json")

# Run
run_colab_flow()

