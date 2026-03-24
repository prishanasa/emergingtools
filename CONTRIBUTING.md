# Contributing to EduEval AI

Thank you for your interest in contributing!

## How to Run Locally
1. Clone the repository
2. Create a conda environment: `conda create -n evaluator python=3.11 -y`
3. Activate it: `conda activate evaluator`
4. Install FAISS: `conda install -c conda-forge faiss-cpu -y`
5. Install dependencies: `pip install -r requirements.txt`
6. Add your Groq API key in the sidebar
7. Run: `streamlit run app.py`

## Project Structure
- `app.py` - Main Streamlit UI
- `rag_engine.py` - FAISS vector DB and RAG pipeline
- `evaluator.py` - Groq LLM evaluation logic
- `utils.py` - Helper functions
- `error_handler.py` - Error handling utilities
- `config.py` - Configuration constants
- `logger.py` - Evaluation logging

## Guidelines
- Keep code clean and well commented
- Test your changes before submitting
- Follow existing code style
