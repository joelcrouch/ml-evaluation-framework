# import argparse
# import sys
# import os
# from sqlalchemy.orm import Session
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Add project root to path to allow imports
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from ml_eval.database.connection import get_db
# from ml_eval.database.models import ModelRun, Evaluation, Response, TestPrompt

# def load_training_history(model_name: str):
#     """Load training history if it exists."""
#     history_path = os.path.join("models", f"{model_name}_history.npy")
#     if os.path.exists(history_path):
#         try:
#             history_dict = np.load(history_path, allow_pickle=True).item()
#             return history_dict
#         except Exception as e:
#             print(f"⚠️  Could not load training history: {e}")
#             return None
#     return None

# def save_performance_charts(evaluations_data: list, run_id: int, model_name: str, training_history=None):
#     """Creates and saves multiple charts for time series model performance."""
    
#     # Extract metrics
#     mse_values = []
#     mae_values = []
#     rmse_values = []
#     scores = []
    
#     for eval_data in evaluations_data:
#         metrics = eval_data['metrics']
#         mse_values.append(metrics.get('mse', 0))
#         mae_values.append(metrics.get('mae', 0))
#         rmse_values.append(metrics.get('rmse', 0))
#         scores.append(eval_data['score'])
    
#     # Determine grid size based on whether we have training history
#     if training_history:
#         fig, axes = plt.subplots(3, 2, figsize=(15, 15))
#     else:
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
#     fig.suptitle(f'Performance Report for {model_name} (Run ID: {run_id})', fontsize=16, fontweight='bold')
    
#     # 1. MSE Distribution
#     ax1 = axes[0, 0]
#     ax1.hist(mse_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
#     ax1.axvline(np.mean(mse_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mse_values):.4f}')
#     ax1.axvline(np.median(mse_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(mse_values):.4f}')
#     ax1.set_xlabel('Mean Squared Error (MSE)')
#     ax1.set_ylabel('Frequency')
#     ax1.set_title('Test Set MSE Distribution')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # 2. MAE Distribution
#     ax2 = axes[0, 1]
#     ax2.hist(mae_values, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
#     ax2.axvline(np.mean(mae_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mae_values):.4f}')
#     ax2.axvline(np.median(mae_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(mae_values):.4f}')
#     ax2.set_xlabel('Mean Absolute Error (MAE)')
#     ax2.set_ylabel('Frequency')
#     ax2.set_title('Test Set MAE Distribution')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     # 3. Score Distribution
#     ax3 = axes[1, 0]
#     ax3.hist(scores, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
#     ax3.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
#     ax3.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.4f}')
#     ax3.set_xlabel('Normalized Score')
#     ax3.set_ylabel('Frequency')
#     ax3.set_title('Test Set Score Distribution (1.0 = Perfect)')
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)
    
#     # 4. Error Metrics Comparison (Box Plot)
#     ax4 = axes[1, 1]
#     data_to_plot = [mse_values, mae_values, rmse_values]
#     bp = ax4.boxplot(data_to_plot, labels=['MSE', 'MAE', 'RMSE'], patch_artist=True)
#     for patch, color in zip(bp['boxes'], ['skyblue', 'lightcoral', 'lightyellow']):
#         patch.set_facecolor(color)
#     ax4.set_ylabel('Error Value')
#     ax4.set_title('Test Set Error Metrics Box Plot')
#     ax4.grid(True, alpha=0.3)
    
#     # 5. Training History - Loss (if available)
#     if training_history:
#         ax5 = axes[2, 0]
#         ax5.plot(training_history['loss'], label='Training Loss', linewidth=2, color='blue')
#         ax5.plot(training_history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
#         ax5.set_xlabel('Epoch')
#         ax5.set_ylabel('Loss (MSE)')
#         ax5.set_title('Training History - Loss')
#         ax5.legend()
#         ax5.grid(True, alpha=0.3)
        
#         # 6. Training History - MAE (if available)
#         ax6 = axes[2, 1]
#         ax6.plot(training_history['mean_absolute_error'], label='Training MAE', linewidth=2, color='blue')
#         ax6.plot(training_history['val_mean_absolute_error'], label='Validation MAE', linewidth=2, color='orange')
#         ax6.set_xlabel('Epoch')
#         ax6.set_ylabel('Mean Absolute Error')
#         ax6.set_title('Training History - MAE')
#         ax6.legend()
#         ax6.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     # Save the figure
#     chart_path = f"reports/run_{run_id}_time_series_report.png"
#     os.makedirs("reports", exist_ok=True)
#     plt.savefig(chart_path, dpi=300, bbox_inches='tight')
#     print(f"\n✅ Performance charts saved to: {chart_path}")
#     plt.close()

# def save_prediction_samples(evaluations_data: list, run_id: int, model_name: str, num_samples=5):
#     """Visualizes actual predictions vs expected values for sample test cases."""
    
#     # Select random samples
#     num_samples = min(num_samples, len(evaluations_data))
#     if num_samples == 0:
#         print("⚠️  No evaluation data to sample from for prediction plots.")
#         return
        
#     sample_indices = np.random.choice(len(evaluations_data), num_samples, replace=False)
    
#     fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
#     if num_samples == 1:
#         axes = [axes]
    
#     fig.suptitle(f'Sample Predictions - {model_name} (Run ID: {run_id})', fontsize=16, fontweight='bold')
    
#     for idx, sample_idx in enumerate(sample_indices):
#         eval_data = evaluations_data[sample_idx]
        
#         # --- FIX START ---
#         # Extract numerical values from prediction, which is likely a list of lists (e.g., [[0.9], [0.8]])
#         predictions = np.array([p[0] for p in eval_data['prediction']]).flatten()

#         # Extract numerical values from ground_truth, which may be a dict or a list of lists
#         raw_gt = eval_data['ground_truth']
#         if isinstance(raw_gt, dict):
#             # If it's a dict, assume the data is in the first value, e.g., {'value': [[0.9]]}
#             gt_list = next(iter(raw_gt.values()))
#         else:
#             gt_list = raw_gt
#         ground_truth = np.array([g[0] for g in gt_list]).flatten()
#         # --- FIX END ---
        
#         ax = axes[idx]
#         x = np.arange(len(predictions))
        
#         ax.plot(x, ground_truth, 'o-', label='Ground Truth', color='blue', linewidth=2, markersize=8)
#         ax.plot(x, predictions, 's--', label='Prediction', color='red', linewidth=2, markersize=8, alpha=0.7)
        
#         # Add error bars
#         errors = np.abs(predictions - ground_truth)
#         ax.fill_between(x, predictions - errors, predictions + errors, alpha=0.2, color='gray')
        
#         ax.set_xlabel('Time Step')
#         ax.set_ylabel('Value')
#         ax.set_title(f'Sample {idx + 1} - MSE: {eval_data["metrics"]["mse"]:.4f}, Score: {eval_data["score"]:.4f}')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     # Save the figure
#     samples_path = f"reports/run_{run_id}_prediction_samples.png"
#     plt.savefig(samples_path, dpi=300, bbox_inches='tight')
#     print(f"✅ Prediction samples saved to: {samples_path}")
#     plt.close()

# def main():
#     """
#     Generates a performance report for a time series model run.
#     """
#     parser = argparse.ArgumentParser(description="Generate a time series performance report for a given ModelRun ID.")
#     parser.add_argument("run_id", type=int, help="The ID of the ModelRun to report on.")
#     parser.add_argument("--samples", type=int, default=5, help="Number of prediction samples to visualize (default: 5)")
#     args = parser.parse_args()

#     run_id = args.run_id

#     print(f"--- Generating Time Series Report for ModelRun ID: {run_id} ---")

#     db: Session = next(get_db())

#     # 1. Fetch the ModelRun
#     model_run = db.query(ModelRun).filter(ModelRun.id == run_id).first()
#     if not model_run:
#         print(f"❌ Error: ModelRun with ID {run_id} not found.")
#         return

#     # 2. Query for all evaluation results
#     results = (
#         db.query(Evaluation, Response, TestPrompt)
#         .join(Response, Evaluation.response_id == Response.id)
#         .join(TestPrompt, Response.test_case_id == TestPrompt.id)
#         .filter(Response.run_id == run_id)
#         .all()
#     )

#     if not results:
#         print(f"❌ Error: No evaluation results found for ModelRun ID: {run_id}.")
#         return

#     # 3. Process the results
#     total_evals = len(results)
#     passed_evals = 0
#     evaluations_data = []
    
#     all_mse = []
#     all_mae = []
#     all_rmse = []
#     all_scores = []

#     for eval_result, response, test_case in results:
#         metrics = eval_result.metrics or {}
        
#         mse = metrics.get('mse', 0)
#         mae = metrics.get('mae', 0)
#         rmse = metrics.get('rmse', 0)
#         score = eval_result.score or 0
        
#         all_mse.append(mse)
#         all_mae.append(mae)
#         all_rmse.append(rmse)
#         all_scores.append(score)
        
#         if eval_result.passed:
#             passed_evals += 1
        
#         # Store evaluation data for visualization
#         evaluations_data.append({
#             'test_case_id': test_case.id,
#             'prediction': response.output_data.get('prediction', []),
#             'ground_truth': test_case.ground_truth,
#             'metrics': metrics,
#             'score': score,
#             'passed': eval_result.passed,
#             'feedback': eval_result.feedback
#         })
    
#     # Check if we have valid data
#     if all(v == 0 for v in all_mse):
#         print("\n⚠️  WARNING: All metrics are zero!")
#         print("This usually means:")
#         print("  1. Test prompts don't have proper 'expected_output' or 'ground_truth' fields")
#         print("  2. Model predictions are not in the expected format")
#         print("  3. Evaluator failed to extract data properly")
#         print("\nSample evaluation feedback:")
#         for i, eval_data in enumerate(evaluations_data[:3]):
#             print(f"  Test Case {eval_data['test_case_id']}: {eval_data['feedback']}")
#         return

#     # 4. Calculate summary statistics
#     pass_rate = (passed_evals / total_evals) * 100 if total_evals > 0 else 0
    
#     mean_mse = np.mean(all_mse)
#     median_mse = np.median(all_mse)
#     std_mse = np.std(all_mse)
    
#     mean_mae = np.mean(all_mae)
#     median_mae = np.median(all_mae)
#     std_mae = np.std(all_mae)
    
#     mean_rmse = np.mean(all_rmse)
#     median_rmse = np.median(all_rmse)
#     std_rmse = np.std(all_rmse)
    
#     mean_score = np.mean(all_scores)
#     median_score = np.median(all_scores)
#     std_score = np.std(all_scores)

#     # 5. Print the report
#     print("\n" + "="*70)
#     print(f"  TIME SERIES PERFORMANCE REPORT")
#     print(f"  Model: {model_run.model_name} (v{model_run.model_version})")
#     print(f"  Run ID: {model_run.id}")
#     print(f"  Model Type: {model_run.model_type}")
#     print("="*70 + "\n")

#     print(f"  📊 Overall Statistics:")
#     print(f"     Total Test Cases: {total_evals}")
#     print(f"     Passed: {passed_evals} ({pass_rate:.2f}%)")
#     print(f"     Failed: {total_evals - passed_evals} ({100 - pass_rate:.2f}%)")
    
#     print(f"\n  📈 Error Metrics:")
#     print(f"     Mean Squared Error (MSE):")
#     print(f"        Mean:   {mean_mse:.6f}")
#     print(f"        Median: {median_mse:.6f}")
#     print(f"        Std:    {std_mse:.6f}")
#     print(f"        Min:    {min(all_mse):.6f}")
#     print(f"        Max:    {max(all_mse):.6f}")
    
#     print(f"\n     Mean Absolute Error (MAE):")
#     print(f"        Mean:   {mean_mae:.6f}")
#     print(f"        Median: {median_mae:.6f}")
#     print(f"        Std:    {std_mae:.6f}")
#     print(f"        Min:    {min(all_mae):.6f}")
#     print(f"        Max:    {max(all_mae):.6f}")
    
#     print(f"\n     Root Mean Squared Error (RMSE):")
#     print(f"        Mean:   {mean_rmse:.6f}")
#     print(f"        Median: {median_rmse:.6f}")
#     print(f"        Std:    {std_rmse:.6f}")
#     print(f"        Min:    {min(all_rmse):.6f}")
#     print(f"        Max:    {max(all_rmse):.6f}")
    
#     print(f"\n  🎯 Normalized Score (1.0 = Perfect):")
#     print(f"     Mean:   {mean_score:.6f}")
#     print(f"     Median: {median_score:.6f}")
#     print(f"     Std:    {std_score:.6f}")
#     print(f"     Min:    {min(all_scores):.6f}")
#     print(f"     Max:    {max(all_scores):.6f}")

#     # 6. Identify best and worst predictions
#     print(f"\n  🏆 Best Predictions (Lowest MSE):")
#     sorted_by_mse = sorted(evaluations_data, key=lambda x: x['metrics'].get('mse', float('inf')))
#     for i, eval_data in enumerate(sorted_by_mse[:3]):
#         mse_val = eval_data['metrics'].get('mse', 0)
#         print(f"     {i+1}. Test Case {eval_data['test_case_id']}: MSE={mse_val:.6f}, Score={eval_data['score']:.4f}")
    
#     print(f"\n  ⚠️  Worst Predictions (Highest MSE):")
#     for i, eval_data in enumerate(sorted_by_mse[-3:]):
#         mse_val = eval_data['metrics'].get('mse', 0)
#         print(f"     {i+1}. Test Case {eval_data['test_case_id']}: MSE={mse_val:.6f}, Score={eval_data['score']:.4f}")

#     # 6. Generate visualizations
#     print(f"\n  📊 Generating visualizations...")
    
#     # Load training history if available
#     training_history = load_training_history(model_run.model_name)
#     if training_history:
#         print(f"  ✅ Training history loaded - including training curves in report")
#     else:
#         print(f"  ℹ️  No training history found - generating test-only charts")
    
#     save_performance_charts(evaluations_data, run_id, model_run.model_name, training_history)
#     save_prediction_samples(evaluations_data, run_id, model_run.model_name, num_samples=args.samples)

#     # 8. Save summary to CSV
#     summary_df = pd.DataFrame({
#         'test_case_id': [e['test_case_id'] for e in evaluations_data],
#         'mse': [e['metrics'].get('mse', 0) for e in evaluations_data],
#         'mae': [e['metrics'].get('mae', 0) for e in evaluations_data],
#         'rmse': [e['metrics'].get('rmse', 0) for e in evaluations_data],
#         'score': [e['score'] for e in evaluations_data],
#         'passed': [e['passed'] for e in evaluations_data]
#     })
    
#     csv_path = f"reports/run_{run_id}_summary.csv"
#     summary_df.to_csv(csv_path, index=False)
#     print(f"  ✅ Summary CSV saved to: {csv_path}")

#     print("\n" + "="*70)
#     print("  Report Generation Complete")
#     print("="*70)


# if __name__ == "__main__":
#     main()

import argparse
import sys
import os
from sqlalchemy.orm import Session
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_eval.database.connection import get_db
from ml_eval.database.models import ModelRun, Evaluation, Response, TestPrompt

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

def save_performance_charts(evaluations_data: list, run_id: int, model_name: str, training_history=None):
    """Creates and saves multiple charts for time series model performance."""
    
    # Extract metrics
    mse_values = []
    mae_values = []
    rmse_values = []
    scores = []
    
    for eval_data in evaluations_data:
        metrics = eval_data['metrics']
        mse_values.append(metrics.get('mse', 0))
        mae_values.append(metrics.get('mae', 0))
        rmse_values.append(metrics.get('rmse', 0))
        scores.append(eval_data['score'])
    
    # Determine grid size based on whether we have training history
    if training_history:
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
    fig.suptitle(f'Performance Report for {model_name} (Run ID: {run_id})', fontsize=16, fontweight='bold')
    
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
    bp = ax4.boxplot(data_to_plot, labels=['MSE', 'MAE', 'RMSE'], patch_artist=True)
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
    
    # Save the figure
    chart_path = f"reports/run_{run_id}_time_series_report.png"
    os.makedirs("reports", exist_ok=True)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Performance charts saved to: {chart_path}")
    plt.close()

def save_prediction_samples(evaluations_data: list, run_id: int, model_name: str, num_samples=5):
    """Visualizes predictions vs labels in the TensorFlow tutorial style."""
    
    # Select random samples
    num_samples = min(num_samples, len(evaluations_data))
    sample_indices = np.random.choice(len(evaluations_data), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    fig.suptitle(f'Sample Predictions (TensorFlow Tutorial Style) - {model_name} (Run ID: {run_id})', 
                 fontsize=16, fontweight='bold')
    
    for idx, sample_idx in enumerate(sample_indices):
        eval_data = evaluations_data[sample_idx]
        
        # Extract predictions and ground truth
        # Prediction is shape (1, 1, 1) -> extract the value
        prediction_raw = eval_data.get('prediction', [])
        ground_truth_raw = eval_data.get('ground_truth', {})
        
        # Handle different formats
        if isinstance(prediction_raw, list) and len(prediction_raw) > 0:
            if isinstance(prediction_raw[0], list):
                predictions = np.array(prediction_raw).flatten()
            else:
                predictions = np.array(prediction_raw)
        else:
            predictions = np.array([0])
        
        # Ground truth from test case
        if isinstance(ground_truth_raw, dict) and 'prediction' in ground_truth_raw:
            ground_truth = np.array(ground_truth_raw['prediction']).flatten()
        else:
            ground_truth = np.array([0])
        
        ax = axes[idx]
        
        # Create time indices
        # Input is at time 0, prediction is at time 1
        input_indices = [0]
        label_indices = [1]
        
        # For baseline: input and prediction should be very similar
        # Plot input value (this is what model saw)
        if len(predictions) > 0:
            ax.plot(input_indices, predictions[:1], 'o', 
                   label='Input', color='blue', markersize=10, zorder=10)
        
        # Plot actual label (ground truth)
        if len(ground_truth) > 0:
            ax.scatter(label_indices, ground_truth[:1], 
                      edgecolors='k', label='Labels', 
                      s=64, zorder=20, color='green')
        
        # Plot predictions
        if len(predictions) > 0:
            ax.scatter(label_indices, predictions[:1], 
                      marker='X', edgecolors='k', label='Predictions',
                      s=64, zorder=30, color='red')
        
        mse = eval_data['metrics'].get('mse', 0)
        score = eval_data.get('score', 0)
        
        ax.set_ylabel('Temperature [normed]')
        ax.set_xlabel('Time [h]')
        ax.set_title(f'Sample {idx + 1} - MSE: {mse:.6f}, Score: {score:.4f}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 1.5)
    
    plt.tight_layout()
    
    # Save the figure
    samples_path = f"reports/run_{run_id}_prediction_samples.png"
    plt.savefig(samples_path, dpi=300, bbox_inches='tight')
    print(f"✅ Prediction samples saved to: {samples_path}")
    plt.close()
    
    
def save_tutorial_style_comparison(evaluations_data: list, run_id: int, model_name: str):
    """Creates a comparison chart exactly like the TensorFlow tutorial."""
    
    # Take first 3 examples
    num_plots = min(3, len(evaluations_data))
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 8))
    if num_plots == 1:
        axes = [axes]
    
    fig.suptitle(f'Baseline Model Performance (Tutorial Style) - {model_name}', 
                 fontsize=16, fontweight='bold')
    
    for n in range(num_plots):
        eval_data = evaluations_data[n]
        ax = axes[n]
        
        # Get input, labels, predictions
        input_data = eval_data.get('input_window', None)
        ground_truth = eval_data.get('ground_truth', {}).get('prediction', [[0]])
        predictions = eval_data.get('prediction', [[[0]]])
        
        # Flatten arrays
        ground_truth_val = np.array(ground_truth).flatten()[0] if len(np.array(ground_truth).flatten()) > 0 else 0
        prediction_val = np.array(predictions).flatten()[0] if len(np.array(predictions).flatten()) > 0 else 0
        
        # Time indices
        input_indices = [0]
        label_indices = [1]
        
        # Plot - Tutorial style
        # Input value as a point
        ax.plot(input_indices, [prediction_val], 
               marker='.', color='blue', markersize=15, 
               label='Inputs', zorder=-10)
        
        # Labels (ground truth)
        ax.scatter(label_indices, [ground_truth_val],
                  edgecolors='k', label='Labels',
                  s=64, c='green', zorder=20)
        
        # Predictions
        ax.scatter(label_indices, [prediction_val],
                  marker='X', edgecolors='k', label='Predictions',
                  s=64, c='red', zorder=30)
        
        ax.set_ylabel('T (degC) [normed]')
        if n == 0:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['t=0 (input)', 't=1 (prediction)'])
    
    axes[-1].set_xlabel('Time [h]')
    plt.tight_layout()
    
    # Save
    tutorial_path = f"reports/run_{run_id}_tutorial_style.png"
    plt.savefig(tutorial_path, dpi=300, bbox_inches='tight')
    print(f"✅ Tutorial-style comparison saved to: {tutorial_path}")
    plt.close()

def main():
    """
    Generates a performance report for a time series model run.
    """
    parser = argparse.ArgumentParser(description="Generate a time series performance report for a given ModelRun ID.")
    parser.add_argument("run_id", type=int, help="The ID of the ModelRun to report on.")
    parser.add_argument("--samples", type=int, default=5, help="Number of prediction samples to visualize (default: 5)")
    args = parser.parse_args()

    run_id = args.run_id

    print(f"--- Generating Time Series Report for ModelRun ID: {run_id} ---")

    db: Session = next(get_db())

    # 1. Fetch the ModelRun
    model_run = db.query(ModelRun).filter(ModelRun.id == run_id).first()
    if not model_run:
        print(f"❌ Error: ModelRun with ID {run_id} not found.")
        return

    # 2. Query for all evaluation results
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
    passed_evals = 0
    evaluations_data = []
    
    all_mse = []
    all_mae = []
    all_rmse = []
    all_scores = []

    for eval_result, response, test_case in results:
        metrics = eval_result.metrics or {}
        
        mse = metrics.get('mse', 0)
        mae = metrics.get('mae', 0)
        rmse = metrics.get('rmse', 0)
        score = eval_result.score or 0
        
        all_mse.append(mse)
        all_mae.append(mae)
        all_rmse.append(rmse)
        all_scores.append(score)
        
        if eval_result.passed:
            passed_evals += 1
        
        # Store evaluation data for visualization
        evaluations_data.append({
            'test_case_id': test_case.id,
            'prediction': response.output_data.get('prediction', []),
            'ground_truth': test_case.ground_truth,
            'metrics': metrics,
            'score': score,
            'passed': eval_result.passed,
            'feedback': eval_result.feedback
        })
    
    # Check if we have valid data
    if all(v == 0 for v in all_mse):
        print("\n⚠️  WARNING: All metrics are zero!")
        print("This usually means:")
        print("  1. Test prompts don't have proper 'expected_output' or 'ground_truth' fields")
        print("  2. Model predictions are not in the expected format")
        print("  3. Evaluator failed to extract data properly")
        print("\nSample evaluation feedback:")
        for i, eval_data in enumerate(evaluations_data[:3]):
            print(f"  Test Case {eval_data['test_case_id']}: {eval_data['feedback']}")
        return

    # 4. Calculate summary statistics
    pass_rate = (passed_evals / total_evals) * 100 if total_evals > 0 else 0
    
    mean_mse = np.mean(all_mse)
    median_mse = np.median(all_mse)
    std_mse = np.std(all_mse)
    
    mean_mae = np.mean(all_mae)
    median_mae = np.median(all_mae)
    std_mae = np.std(all_mae)
    
    mean_rmse = np.mean(all_rmse)
    median_rmse = np.median(all_rmse)
    std_rmse = np.std(all_rmse)
    
    mean_score = np.mean(all_scores)
    median_score = np.median(all_scores)
    std_score = np.std(all_scores)

    # 5. Print the report
    print("\n" + "="*70)
    print(f"  TIME SERIES PERFORMANCE REPORT")
    print(f"  Model: {model_run.model_name} (v{model_run.model_version})")
    print(f"  Run ID: {model_run.id}")
    print(f"  Model Type: {model_run.model_type}")
    print("="*70 + "\n")

    print(f"  📊 Overall Statistics:")
    print(f"     Total Test Cases: {total_evals}")
    print(f"     Passed: {passed_evals} ({pass_rate:.2f}%)")
    print(f"     Failed: {total_evals - passed_evals} ({100 - pass_rate:.2f}%)")
    
    print(f"\n  📈 Error Metrics:")
    print(f"     Mean Squared Error (MSE):")
    print(f"        Mean:   {mean_mse:.6f}")
    print(f"        Median: {median_mse:.6f}")
    print(f"        Std:    {std_mse:.6f}")
    print(f"        Min:    {min(all_mse):.6f}")
    print(f"        Max:    {max(all_mse):.6f}")
    
    print(f"\n     Mean Absolute Error (MAE):")
    print(f"        Mean:   {mean_mae:.6f}")
    print(f"        Median: {median_mae:.6f}")
    print(f"        Std:    {std_mae:.6f}")
    print(f"        Min:    {min(all_mae):.6f}")
    print(f"        Max:    {max(all_mae):.6f}")
    
    print(f"\n     Root Mean Squared Error (RMSE):")
    print(f"        Mean:   {mean_rmse:.6f}")
    print(f"        Median: {median_rmse:.6f}")
    print(f"        Std:    {std_rmse:.6f}")
    print(f"        Min:    {min(all_rmse):.6f}")
    print(f"        Max:    {max(all_rmse):.6f}")
    
    print(f"\n  🎯 Normalized Score (1.0 = Perfect):")
    print(f"     Mean:   {mean_score:.6f}")
    print(f"     Median: {median_score:.6f}")
    print(f"     Std:    {std_score:.6f}")
    print(f"     Min:    {min(all_scores):.6f}")
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

    # 7. Generate visualizations
    print(f"\n  📊 Generating visualizations...")
    
    # Load training history if available
    training_history = load_training_history(model_run.model_name)
    if training_history:
        print(f"  ✅ Training history loaded - including training curves in report")
    else:
        print(f"  ℹ️  No training history found - generating test-only charts")
    
    save_performance_charts(evaluations_data, run_id, model_run.model_name, training_history)
    save_prediction_samples(evaluations_data, run_id, model_run.model_name, num_samples=args.samples)
    save_tutorial_style_comparison(evaluations_data, run_id, model_run.model_name)

    # 8. Save summary to CSV
    summary_df = pd.DataFrame({
        'test_case_id': [e['test_case_id'] for e in evaluations_data],
        'mse': [e['metrics'].get('mse', 0) for e in evaluations_data],
        'mae': [e['metrics'].get('mae', 0) for e in evaluations_data],
        'rmse': [e['metrics'].get('rmse', 0) for e in evaluations_data],
        'score': [e['score'] for e in evaluations_data],
        'passed': [e['passed'] for e in evaluations_data]
    })
    
    csv_path = f"reports/run_{run_id}_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  ✅ Summary CSV saved to: {csv_path}")

    print("\n" + "="*70)
    print("  Report Generation Complete")
    print("="*70)


if __name__ == "__main__":
    main()