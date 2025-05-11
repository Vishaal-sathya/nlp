# NLP Documentation Model Evaluation

This project contains tools to evaluate the performance of the NLP documentation generator model using standard NLP metrics.

## Overview

The evaluation script (`test_documentation_model.py`) tests the model's ability to generate documentation for code snippets by comparing the generated documentation against reference (human-written) documentation using three standard metrics:

- **BLEU (Bilingual Evaluation Understudy)**: Measures precision of n-grams between generated and reference text
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Measures recall of n-grams
- **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**: Considers synonyms and stemming for a more semantic evaluation

## Test Dataset

The evaluation uses a dataset of 10 code-documentation pairs covering various Python programming patterns:

1. Simple function with docstring
2. Class with methods
3. Function with type hints
4. Recursive function
5. Function with error handling
6. Function with list comprehension
7. Function with default parameters
8. Class with inheritance
9. Function with dictionary operations
10. Decorator function

## How to Run

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the evaluation script:
   ```
   python test_documentation_model.py
   ```

3. The script will:
   - Load the model specified in your configuration
   - Generate documentation for each code snippet
   - Calculate BLEU, ROUGE, and METEOR scores for each pair
   - Print detailed results for each test case
   - Calculate and print average scores
   - Save all results to `model_evaluation_results.json`

## Interpreting Results

### BLEU Score
- Range: 0 to 1 (higher is better)
- Measures how many n-grams in the generated text match the reference
- Scores above 0.5 generally indicate good quality

### ROUGE Scores
- Range: 0 to 1 (higher is better)
- ROUGE-1: Measures unigram (single word) overlap
- ROUGE-2: Measures bigram (two consecutive words) overlap
- ROUGE-L: Measures longest common subsequence

### METEOR Score
- Range: 0 to 1 (higher is better)
- More semantically aware than BLEU
- Considers synonyms and word stems
- Generally, scores above 0.4 indicate good quality

## Customizing the Evaluation

You can modify the test dataset in `test_documentation_model.py` to include different code examples or reference documentation that better match your specific use case.