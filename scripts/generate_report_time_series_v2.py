import argparse
import sys
import os
from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hashlib
import json

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_eval.database.connection import get_db
from ml_eval.database.models import ModelRun, Evaluation, Response, TestPrompt

def generate_report_filename_base(model_run: ModelRun) -> str:
    """Generates a structured, descriptive base filename for a report."""
    # Sanitize model name by replacing underscores
    model_name = model_run.model_name.replace('_', '-')
    version = f"v{model_run.model_version}"
    
    # Create a hash from the config dictionary if it exists
    params_str = ""
    # Ensure config is a dict before processing
    if model_run.config and isinstance(model_run.config, dict):
        # Create a stable, sorted JSON string to ensure consistent hashing
        config_json = json.dumps(model_run.config, sort_keys=True)
        # Use a short SHA1 hash to keep filenames manageable
        params_hash = hashlib.sha1(config_json.encode()).hexdigest()[:8]
        params_str = f"_config-{params_hash}"
        
    run_id_str = f"_run-{model_run.id}"
    
    return f"{model_name}_{version}{params_str}{run_id_str}"

def load_training_history(model_name: str):
    """Load training history if it exists."""
    history_path = os.path.join("models", f"{model_name}_history.npy")
    if os.path.exists(history_path):
        try:
            history_dict = np.load(history_path, allow_pickle=True).item()
            return history_dict
        except Exception as e:
            print(f"⚠️  Could not load training history: {e}")
            return None
    return None

def save_performance_charts(evaluations_data: list, model_run: ModelRun, save_path: str, training_history=None):
    """Creates and saves multiple charts for time series model performance."""
    mse_values = [e['metrics'].get('mse', 0) for e in evaluations_data]
    mae_values = [e['metrics'].get('mae', 0) for e in evaluations_data]
    rmse_values = [e['metrics'].get('rmse', 0) for e in evaluations_data]
    scores = [e['score'] for e in evaluations_data]
    
    grid_rows = 3 if training_history else 2
    fig, axes = plt.subplots(grid_rows, 2, figsize=(15, 5 * grid_rows))
    # Ensure axes is always a 2D array for consistent indexing
    if grid_rows == 1: # If only 1 row, axes is a 1D array or single Axes object
        if isinstance(axes, np.ndarray) and axes.ndim == 1:
            axes = axes.reshape(1, -1) # Make it 2D
        else: # single Axes object
            axes = np.array([axes]).reshape(1, -1)
    
    fig.suptitle(f'Performance Report for {model_run.model_name} (Run ID: {model_run.id})', fontsize=16, fontweight='bold')
    
    # 1. MSE Distribution
    ax1 = axes[0, 0]
    ax1.hist(mse_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(mse_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mse_values):.4f}')
    ax1.axvline(np.median(mse_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(mse_values):.4f}')
    ax1.set_xlabel('Mean Squared Error (MSE)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Test Set MSE Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE Distribution
    ax2 = axes[0, 1]
    ax2.hist(mae_values, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(mae_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mae_values):.4f}')
    ax2.axvline(np.median(mae_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(mae_values):.4f}')
    ax2.set_xlabel('Mean Absolute Error (MAE)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Test Set MAE Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Score Distribution
    ax3 = axes[1, 0]
    ax3.hist(scores, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
    ax3.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.4f}')
    ax3.set_xlabel('Normalized Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Test Set Score Distribution (1.0 = Perfect)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Metrics Comparison (Box Plot)
    ax4 = axes[1, 1]
    data_to_plot = [mse_values, mae_values, rmse_values]
    # MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9
    bp = ax4.boxplot(data_to_plot, tick_labels=['MSE', 'MAE', 'RMSE'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['skyblue', 'lightcoral', 'lightyellow']):
        patch.set_facecolor(color)
    ax4.set_ylabel('Error Value')
    ax4.set_title('Test Set Error Metrics Box Plot')
    ax4.grid(True, alpha=0.3)
    
    # 5. Training History - Loss (if available)
    if training_history:
        ax5 = axes[2, 0]
        ax5.plot(training_history['loss'], label='Training Loss', linewidth=2, color='blue')
        ax5.plot(training_history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss (MSE)')
        ax5.set_title('Training History - Loss')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Training History - MAE (if available)
        ax6 = axes[2, 1]
        ax6.plot(training_history['mean_absolute_error'], label='Training MAE', linewidth=2, color='blue')
        ax6.plot(training_history['val_mean_absolute_error'], label='Validation MAE', linewidth=2, color='orange')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Mean Absolute Error')
        ax6.set_title('Training History - MAE')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Performance charts saved to: {save_path}")
    plt.close()

def save_prediction_samples(evaluations_data: list, model_run: ModelRun, save_path: str, num_samples=5):
    """Visualizes actual predictions vs expected values for sample test cases."""
    num_samples = min(num_samples, len(evaluations_data))
    if num_samples == 0:
        print("⚠️  No evaluation data to sample from for prediction plots.")
        return
        
    sample_indices = np.random.choice(len(evaluations_data), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples), squeeze=False)
    
    fig.suptitle(f'Sample Predictions - {model_run.model_name} (Run ID: {model_run.id})', fontsize=16, fontweight='bold')
    
    for idx, sample_idx in enumerate(sample_indices):
        eval_data = evaluations_data[sample_idx]
        
        # Extract numerical values from prediction, which is likely a list of lists (e.g., [[0.9], [0.8]])
        predictions = np.array([p[0] for p in eval_data['prediction']]).flatten()

        # Extract numerical values from ground_truth, which may be a dict or a list of lists
        raw_gt = eval_data['ground_truth']
        if isinstance(raw_gt, dict):
            # If it's a dict, assume the data is in the first value, e.g., {'value': [[0.9]]}
            gt_list = next(iter(raw_gt.values()))
        else:
            gt_list = raw_gt
        ground_truth = np.array([g[0] for g in gt_list]).flatten()
        
        ax = axes[idx, 0] # Use [idx, 0] for consistent 2D indexing
        x = np.arange(len(predictions))
        
        ax.plot(x, ground_truth, 'o-', label='Ground Truth', color='blue', linewidth=2, markersize=8)
        ax.plot(x, predictions, 's--', label='Prediction', color='red', linewidth=2, markersize=8, alpha=0.7)
        
        # Add error bars
        errors = np.abs(predictions - ground_truth)
        ax.fill_between(x, predictions - errors, predictions + errors, alpha=0.2, color='gray')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Sample {idx + 1} - MSE: {eval_data["metrics"]["mse"]:.4f}, Score: {eval_data["score"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Prediction samples saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate a time series performance report.")
    parser.add_argument("run_id", type=int, help="The ID of the ModelRun to report on.")
    parser.add_argument("--samples", type=int, default=5, help="Number of prediction samples to visualize.")
    args = parser.parse_args()

    print(f"--- Generating Time Series Report for ModelRun ID: {args.run_id} ---")

    db: Session = next(get_db())

    model_run = db.query(ModelRun).filter(ModelRun.id == args.run_id).first()
    if not model_run:
        print(f"❌ Error: ModelRun with ID {args.run_id} not found.")
        return

    results = db.query(Evaluation, Response, TestPrompt).join(Response, Evaluation.response_id == Response.id).join(TestPrompt, Response.test_case_id == TestPrompt.id).filter(Response.run_id == args.run_id).all()
    if not results:
        print(f"❌ Error: No evaluation results found for ModelRun ID: {args.run_id}.")
        return

    # --- NEW: Generate base filename ---
    base_filename = generate_report_filename_base(model_run)
    os.makedirs("reports", exist_ok=True)
    
    # --- NEW: Define all report paths ---
    chart_path = f"reports/{base_filename}_performance-charts.png"
    samples_path = f"reports/{base_filename}_prediction-samples.png"
    csv_path = f"reports/{base_filename}_summary-stats.csv"

    # Process results into a list of dicts
    evaluations_data = [
        {
            'test_case_id': tc.id,
            'prediction': resp.output_data.get('prediction', []),
            'ground_truth': tc.ground_truth,
            'metrics': ev.metrics or {},
            'score': ev.score or 0,
            'passed': ev.passed
        } for ev, resp, tc in results
    ]
    
    # 5. Print the report (text-based)
    print("\n" + "="*70)
    print(f"  TIME SERIES PERFORMANCE REPORT")
    print(f"  Model: {model_run.model_name} (v{model_run.model_version})")
    print(f"  Run ID: {model_run.id}")
    print(f"  Model Type: {model_run.model_type}")
    if model_run.config:
        print(f"  Config: {json.dumps(model_run.config)}")
    print("="*70 + "\n")

    total_evals = len(evaluations_data)
    passed_evals = sum(1 for e in evaluations_data if e['passed'])
    pass_rate = (passed_evals / total_evals) * 100 if total_evals > 0 else 0

    all_mse = [e['metrics'].get('mse', 0) for e in evaluations_data]
    all_mae = [e['metrics'].get('mae', 0) for e in evaluations_data]
    all_rmse = [e['metrics'].get('rmse', 0) for e in evaluations_data]
    all_scores = [e['score'] for e in evaluations_data]

    # ... (rest of the print statements for summary statistics - unchanged from previous versions) ...
    # This block is quite long, so just imagine it's here
    # 📊 Overall Statistics:
    print(f"  📊 Overall Statistics:")
    print(f"     Total Test Cases: {total_evals}")
    print(f"     Passed: {passed_evals} ({pass_rate:.2f}%)")
    print(f"     Failed: {total_evals - passed_evals} ({100 - pass_rate:.2f}%)")
    
    print(f"\n  📈 Error Metrics:")
    print(f"     Mean Squared Error (MSE):")
    print(f"        Mean:   {np.mean(all_mse):.6f}")
    print(f"        Median: {np.median(all_mse):.6f}")
    print(f"        Std:    {np.std(all_mse):.6f}")
    print(f"        Min:    {min(all_mse):.6f}")
    print(f"        Max:    {max(all_mse):.6f}")
    
    print(f"\n     Mean Absolute Error (MAE):")
    print(f"        Mean:   {np.mean(all_mae):.6f}")
    print(f"        Median: {np.median(all_mae):.6f}")
    print(f"        Std:    {np.std(all_mae):.6f}")
    print(f"        Min:    {min(all_mae):.6f}")
    print(f"        Max:    {max(all_mae):.6f}")
    
    print(f"\n     Root Mean Squared Error (RMSE):")
    print(f"        Mean:   {np.mean(all_rmse):.6f}")
    print(f"        Median: {np.median(all_rmse):.6f}")
    print(f"        Std:    {np.std(all_rmse):.6f}")
    print(f"        Min:    {min(all_rmse):.6f}")
    print(f"        Max:    {max(all_rmse):.6f}")
    
    print(f"\n  🎯 Normalized Score (1.0 = Perfect):")
    print(f"     Mean:   {np.mean(all_scores):.6f}")
    print(f"        Median: {np.median(all_scores):.6f}")
    print(f"        Std:    {np.std(all_scores):.6f}")
    print(f"        Min:    {min(all_scores):.6f}")
    print(f"     Max:    {max(all_scores):.6f}")

    # 6. Identify best and worst predictions
    print(f"\n  🏆 Best Predictions (Lowest MSE):")
    sorted_by_mse = sorted(evaluations_data, key=lambda x: x['metrics'].get('mse', float('inf')))
    for i, eval_data in enumerate(sorted_by_mse[:3]):
        mse_val = eval_data['metrics'].get('mse', 0)
        print(f"     {i+1}. Test Case {eval_data['test_case_id']}: MSE={mse_val:.6f}, Score={eval_data['score']:.4f}")
    
    print(f"\n  ⚠️  Worst Predictions (Highest MSE):")
    for i, eval_data in enumerate(sorted_by_mse[-3:]):
        mse_val = eval_data['metrics'].get('mse', 0)
        print(f"     {i+1}. Test Case {eval_data['test_case_id']}: MSE={mse_val:.6f}, Score={eval_data['score']:.4f}")

    # --- UPDATED: Generate visualizations with new paths ---
    print(f"\n  📊 Generating visualizations...")
    training_history = load_training_history(model_run.model_name)
    
    if evaluations_data: # Only attempt to save if there's data
        save_performance_charts(evaluations_data, model_run, chart_path, training_history)
        save_prediction_samples(evaluations_data, model_run, samples_path, num_samples=args.samples)
        # Note: save_tutorial_style_comparison is not updated for new filename, keep old for now.
        # save_tutorial_style_comparison(evaluations_data, run_id, model_run.model_name)

    # --- UPDATED: Save summary to CSV with new path ---
    summary_df = pd.DataFrame([
        {
            'test_case_id': e['test_case_id'],
            'mse': e['metrics'].get('mse', 0),
            'mae': e['metrics'].get('mae', 0),
            'rmse': e['metrics'].get('rmse', 0),
            'score': e['score'],
            'passed': e['passed']
        } for e in evaluations_data
    ])
    
    if not summary_df.empty:
        summary_df.to_csv(csv_path, index=False)
        print(f"  ✅ Summary CSV saved to: {csv_path}")

    print("\n" + "="*70)
    print("  Report Generation Complete")
    print("="*70)

if __name__ == "__main__":
    main()
