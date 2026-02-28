from vector_store import retrieve_context
from prompts import build_evaluation_prompt
from sentence_transformers import SentenceTransformer
import numpy as np

# Dummy LLM function (replace with OpenAI / HF later)
def call_llm(prompt):
    """
    Replace this with actual API call:
    - OpenAI GPT
    - HuggingFace model
    """
    return "LLM response based on prompt:\n" + prompt[:500]


# ------------------------------
# Concept Similarity Utility
# ------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def similarity_score(text1, text2):
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])
    sim = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(sim)


# ------------------------------
# Main Evaluation Function
# ------------------------------
def evaluate_answer(question, student_answer):
    # Step 1: Retrieve syllabus-grounded context (RAG)
    context = retrieve_context(question)

    # Step 2: Build prompt for LLM reasoning
    prompt = build_evaluation_prompt(
        question=question,
        student_answer=student_answer,
        context=context
    )

    # Step 3: Call LLM (mock / real)
    llm_feedback = call_llm(prompt)

    # Step 4: Compute simple coverage score
    sim = similarity_score(student_answer, context)

    if sim > 0.7:
        score = 5
    elif sim > 0.5:
        score = 3
    else:
        score = 1

    return {
        "context_used": context,
        "score": score,
        "similarity": sim,
        "feedback": llm_feedback
    }
