# Use a pipeline as a high-level helper
from transformers import pipeline
from config_utils import get_model_name

# Load the code documentation generation pipeline with model name from config
pipe = pipeline("summarization", model=get_model_name())

# Example code snippet
code = """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
"""

# Get the summary
summary = pipe(code)
print("Code Summary:", summary[0]["summary_text"])
