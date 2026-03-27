# Report generation utilities for EduEval AI

import datetime

def generate_student_report(student_name, question, result):
    """Generate a text report for a single student evaluation"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
====================================
EDUEVAL AI - EVALUATION REPORT
====================================
Date        : {timestamp}
Student     : {student_name}
------------------------------------
QUESTION:
{question}
------------------------------------
SCORE       : {result.get('marks_awarded', 0)} / {result.get('max_marks', 10)}
PERCENTAGE  : {result.get('percentage', 0)}%
GRADE       : {result.get('grade', 'N/A')}
------------------------------------
CONCEPTS COVERED:
{', '.join(result.get('concepts_covered', [])) or 'None identified'}

CONCEPTS MISSING:
{', '.join(result.get('concepts_missing', [])) or 'None identified'}
------------------------------------
FEEDBACK:
{result.get('detailed_feedback', 'No feedback available')}
------------------------------------
MODEL ANSWER:
{result.get('improved_answer', 'Not available')}
====================================
"""
    return report.strip()

def save_report(report, filename=None):
    """Save report to a text file"""
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(report)
    return filename
