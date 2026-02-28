from evaluator import evaluate_answer

if __name__ == "__main__":
    question = "Explain deadlock in operating systems."
    student_answer = """
    Deadlock occurs when processes stop executing because they are waiting for each other.
    It causes system freeze.
    """

    result = evaluate_answer(question, student_answer)

    print("\n--- Retrieved Context ---")
    print(result["context_used"])

    print("\n--- Score ---")
    print(result["score"], "/ 5")

    print("\n--- Similarity Score ---")
    print(round(result["similarity"], 3))

    print("\n--- Feedback ---")
    print(result["feedback"])
