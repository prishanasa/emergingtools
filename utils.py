# Utility functions for EduEval AI

def marks_to_grade(marks, max_marks=10):
    """Convert marks to letter grade"""
    pct = (marks / max_marks) * 100
    if pct >= 90: return "A+"
    elif pct >= 80: return "A"
    elif pct >= 70: return "B"
    elif pct >= 50: return "C"
    elif pct >= 30: return "D"
    else: return "F"

def truncate_text(text, max_chars=100):
    """Truncate long text with ellipsis"""
    return text[:max_chars] + "..." if len(text) > max_chars else text

def format_percentage(marks, max_marks=10):
    """Return formatted percentage string"""
    return f"{round((marks / max_marks) * 100, 1)}%"

def get_grade_color(grade):
    """Return hex color for each grade"""
    colors = {
        "A+": "#3fb950", "A": "#3fb950",
        "B": "#58a6ff", "C": "#e3b341",
        "D": "#f0883e", "F": "#f85149"
    }
    return colors.get(grade, "#8b949e")
