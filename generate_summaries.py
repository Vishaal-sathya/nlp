import json
import sys
import os

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required functions
from code_analyzer import load_model, generate_summary
from test_documentation_model import TEST_DATASET

def generate_and_save_summaries():
    """Load the model, generate summaries for test data, and save results to JSON."""
    # Load the model as requested
    print("Loading model...")
    model = load_model()
    
    # Initialize list to store results
    results = []
    
    print("Generating summaries for test data...")
    # Process each test case
    for i, test_case in enumerate(TEST_DATASET):
        code = test_case["code"]
        reference = test_case["reference"]
        
        # Generate summary using the model
        summary = generate_summary(code, model)
        
        # Store the result
        results.append({
            "test_case_id": i + 1,
            "code": code.strip(),
            "reference": reference,
            "generated_summary": summary
        })
        
        # Print progress
        print(f"Generated summary for test case {i+1}/{len(TEST_DATASET)}")
    
    # Save results to JSON file
    output_file = "generated_summaries.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll summaries generated and saved to {output_file}")

if __name__ == "__main__":
    generate_and_save_summaries()