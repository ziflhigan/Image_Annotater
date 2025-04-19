"""Prompt & schema definition used when querying Gemini."""

SYSTEM_PROMPT = """
You are an assistant helping to create bilingual (English & Malay) image–question–answer pairs 
for training a Malaysian Vision–Language Model.  
Reply with JSON that follows EXACTLY this TypeScript interface:

interface GeminiQA {
  task_type: 'captioning' | 'vqa' | 'instruction'; // choose 1
  question_en: string;   // empty string if task_type === 'captioning'
  question_ms: string;   // "                                                "
  answer_en:   string;   // description or answer
  answer_ms:   string;   // Malay translation of answer_en
  difficulty:  'easy' | 'medium' | 'hard';
  language_quality_score: number; // 0‑5 inclusive, float allowed
  tags: string[]; // optional short keywords e.g. ["food", "outdoor"]
}
"""


# The schema will be supplied to Gemini via `response_schema`.
# We keep it in a plain dict so we don't import Pydantic here (faster cold start).

def gemini_response_schema():
    return {
        "task_type": ("literal", ["captioning", "vqa", "instruction"]),
        "question_en": ("str",),
        "question_ms": ("str",),
        "answer_en": ("str",),
        "answer_ms": ("str",),
        "difficulty": ("literal", ["easy", "medium", "hard"]),
        "language_quality_score": ("float",),
        "tags": ("list_str",),
    }
