import argparse
import sys
import os
from sqlalchemy.orm import Session
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_eval.database.connection import get_db
from ml_eval.database.models import ModelRun, Evaluation, Response, TestPrompt

def save_accuracy_chart(category_performance: dict, run_id: int, model_name: str):
    """Saves a bar chart of category accuracies with error bars."""
    
    # Prepare data for plotting
    categories = sorted(category_performance.keys())
    accuracies = [(category_performance[cat]['correct'] / category_performance[cat]['total']) for cat in categories]
    totals = [category_performance[cat]['total'] for cat in categories]
    
    # Calculate standard error for error bars: sqrt(p * (1-p) / n)
    errors = [np.sqrt(acc * (1 - acc) / n) if n > 0 else 0 for acc, n in zip(accuracies, totals)]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(categories, accuracies, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
    
    # Formatting
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Category-wise Accuracy for Run ID: {run_id} ({model_name})')
    ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha="right")
    plt.tight_layout()
    
    # Add accuracy values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2%}', ha='center', va='bottom')

    # Save the figure
    chart_path = f"reports/run_{run_id}_accuracy_report.png"
    plt.savefig(chart_path)
    print(f"\n✅ Chart saved to: {chart_path}")

def main():
    """
    Generates a performance report for a given ModelRun ID.
    """
    parser = argparse.ArgumentParser(description="Generate a performance report for a given ModelRun ID.")
    parser.add_argument("run_id", type=int, help="The ID of the ModelRun to report on.")
    args = parser.parse_args()

    run_id = args.run_id

    print(f"--- Generating Report for ModelRun ID: {run_id} ---")

    db: Session = next(get_db())

    # 1. Fetch the ModelRun to get its name and version
    model_run = db.query(ModelRun).filter(ModelRun.id == run_id).first()
    if not model_run:
        print(f"❌ Error: ModelRun with ID {run_id} not found.")
        return

    # 2. Query for all relevant evaluation data for this run
    results = (
        db.query(Evaluation, Response, TestPrompt)
        .join(Response, Evaluation.response_id == Response.id)
        .join(TestPrompt, Response.test_case_id == TestPrompt.id)
        .filter(Response.run_id == run_id)
        .all()
    )

    if not results:
        print(f"❌ Error: No evaluation results found for ModelRun ID: {run_id}.")
        return

    # 3. Process the results
    total_evals = len(results)
    correct_evals = 0
    failures = []
    category_performance = defaultdict(lambda: {'correct': 0, 'total': 0})

    for eval_result, response, test_case in results:
        is_pass = eval_result.passed
        # Use ground truth label for category, which is the flower type
        category = test_case.ground_truth.get("label", "uncategorized")

        category_performance[category]['total'] += 1
        if is_pass:
            correct_evals += 1
            category_performance[category]['correct'] += 1
        else:
            failures.append({
                "image_path": test_case.input_data.get("image_path", "N/A"),
                "ground_truth": test_case.ground_truth.get("label", "N/A"),
                "prediction": response.output_data.get("predicted_label", "N/A")
            })

    # 4. Print the report
    overall_accuracy = (correct_evals / total_evals) * 100 if total_evals > 0 else 0

    print("\n" + "="*50)
    print(f"  Performance Report for: {model_run.model_name} (v{model_run.model_version})")
    print(f"  Run ID: {model_run.id}")
    print("="*50 + "\n")

    print(f"  Overall Accuracy: {overall_accuracy:.2f}% ({correct_evals}/{total_evals} correct)")
    
    print("\n--- Category Performance ---")
    for category, stats in sorted(category_performance.items()):
        cat_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  - {category.capitalize():<15}: {cat_accuracy:.2f}% ({stats['correct']}/{stats['total']})")

    if failures:
        print("\n--- Analysis of Failures ---")
        for i, failure in enumerate(failures):
            print(f"  {i+1}. Model failed on image: {failure['image_path']}")
            print(f"     - Ground Truth: '{failure['ground_truth']}'")
            print(f"     - Prediction:   '{failure['prediction']}'\n")
    else:
        print("\n--- No failures recorded in this run! ---")

    # 5. Generate and save the chart
    if category_performance:
        save_accuracy_chart(category_performance, run_id, model_run.model_name)

    print("\n" + "="*50)
    print("  Report Complete")
    print("="*50)


if __name__ == "__main__":
    main()
