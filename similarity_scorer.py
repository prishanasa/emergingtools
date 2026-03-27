# Similarity scoring utilities for EduEval AI

def compute_word_overlap(answer, reference):
    """Compute simple word overlap score between answer and reference"""
    answer_words = set(answer.lower().split())
    reference_words = set(reference.lower().split())
    if not reference_words:
        return 0.0
    overlap = answer_words.intersection(reference_words)
    return round(len(overlap) / len(reference_words) * 100, 1)

def estimate_answer_completeness(answer, min_words=30, good_words=80):
    """Estimate how complete an answer is based on word count"""
    word_count = len(answer.split())
    if word_count < min_words:
        return "Too brief"
    elif word_count < good_words:
        return "Adequate"
    else:
        return "Detailed"

def get_answer_stats(answer):
    """Return basic statistics about a student answer"""
    words = answer.split()
    sentences = answer.split('.')
    return {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
        "completeness": estimate_answer_completeness(answer)
    }
