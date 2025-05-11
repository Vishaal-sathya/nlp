import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Load your JSON file
with open("test.json", "r") as f:
    data = json.load(f)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Smoothing for BLEU
smooth_fn = SmoothingFunction().method1

# Store average scores
bleu_scores = []
meteor_scores = []
rouge1_scores = []
rougeL_scores = []

# Evaluate each function
for item in data["functions"]:
    ref = item["provided_docstring"]
    gen = item["generated_docstring"]

    # BLEU
    bleu = sentence_bleu([ref.split()], gen.split(), smoothing_function=smooth_fn)
    bleu_scores.append(bleu)

    # METEOR
    meteor = meteor_score([ref.split()], gen.split())
    meteor_scores.append(meteor)

    # ROUGE
    rouge = scorer.score(ref, gen)
    rouge1_scores.append(rouge["rouge1"].fmeasure)
    rougeL_scores.append(rouge["rougeL"].fmeasure)

# Print average scores
print()
print()
print(f"Average BLEU: {sum(bleu_scores) / len(bleu_scores):.4f}")
print(f"Average METEOR: {sum(meteor_scores) / len(meteor_scores):.4f}")
print(f"Average ROUGE-1: {sum(rouge1_scores) / len(rouge1_scores):.4f}")
print(f"Average ROUGE-L: {sum(rougeL_scores) / len(rougeL_scores):.4f}")
print()

import matplotlib.pyplot as plt

# Calculate average scores
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_meteor = sum(meteor_scores) / len(meteor_scores)
avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

# Prepare data
metrics = ["BLEU", "METEOR", "ROUGE-1", "ROUGE-L"]
scores = [avg_bleu, avg_meteor, avg_rouge1, avg_rougeL]

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, scores, color=["skyblue", "orange", "lightgreen", "salmon"])
plt.ylim(0, 1)
plt.title("Average Evaluation Metrics for Generated Docstrings")
plt.ylabel("Score")
plt.xlabel("Metric")

# Add value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center')

plt.tight_layout()
plt.show()
