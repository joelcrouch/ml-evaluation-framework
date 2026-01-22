import argparse
import sys
import os
from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hashlib
import json
import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_eval.database.connection import get_db
from ml_eval.database.models import ModelRun, Evaluation, Response, TestPrompt

# --- UTILITY CLASSES AND FUNCTIONS ---

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        self.input_width, self.label_width, self.shift = input_width, label_width, shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data, targets=None, sequence_length=self.total_window_size,
            sequence_stride=1, shuffle=False, batch_size=32)
        return ds.map(self.split_window)

    @property
    def test(self): return self.make_dataset(self.test_df)

def generate_report_filename_base(model_run: ModelRun) -> str:
    model_name = model_run.model_name.replace('_', '-')
    version = f"v{model_run.model_version}"
    params_str = ""
    if model_run.config and isinstance(model_run.config, dict):
        config_json = json.dumps(model_run.config, sort_keys=True)
        params_hash = hashlib.sha1(config_json.encode()).hexdigest()[:8]
        params_str = f"_config-{params_hash}"
    run_id_str = f"_run-{model_run.id}"
    return f"{model_name}_{version}{params_str}{run_id_str}"

def load_training_history(model_name: str):
    history_path = os.path.join("models", f"{model_name}_history.npy")
    if os.path.exists(history_path):
        try:
            return np.load(history_path, allow_pickle=True).item()
        except Exception as e:
            print(f"⚠️  Could not load training history: {e}")
    return None

# --- PLOTTING FUNCTIONS ---

def save_performance_charts(evaluations_data: list, model_run: ModelRun, save_path: str, training_history=None):
    print("Generating performance charts...")
    mse_values = [e['metrics'].get('mse', 0) for e in evaluations_data]
    mae_values = [e['metrics'].get('mae', 0) for e in evaluations_data]
    scores = [e['score'] for e in evaluations_data]
    
    grid_rows, grid_cols = (3, 2) if training_history else (2, 2)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 5 * grid_rows))
    axes = np.array(axes).flatten()

    fig.suptitle(f'Performance Report for {model_run.model_name} (Run ID: {model_run.id})', fontsize=16, fontweight='bold')
    
    axes[0].hist(mse_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Test Set MSE Distribution')
    axes[0].set_xlabel('Mean Squared Error (MSE)')
    
    axes[1].hist(mae_values, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_title('Test Set MAE Distribution')
    axes[1].set_xlabel('Mean Absolute Error (MAE)')
    
    axes[2].hist(scores, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[2].set_title('Test Set Score Distribution')
    axes[2].set_xlabel('Normalized Score (1.0 = Perfect)')

    axes[3].boxplot([mse_values, mae_values], tick_labels=['MSE', 'MAE'], patch_artist=True)
    axes[3].set_title('Error Metrics Box Plot')
    
    if training_history:
        axes[4].plot(training_history['loss'], label='Training Loss')
        axes[4].plot(training_history['val_loss'], label='Validation Loss')
        axes[4].set_title('Training History - Loss')
        axes[4].legend()
        axes[5].plot(training_history['mean_absolute_error'], label='Training MAE')
        axes[5].plot(training_history['val_mean_absolute_error'], label='Validation MAE')
        axes[5].set_title('Training History - MAE')
        axes[5].legend()

    for ax in axes: ax.grid(True, alpha=0.3) 
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"✅ Performance charts saved to: {save_path}")
    plt.close()

def save_prediction_samples(evaluations_data: list, model_run: ModelRun, save_path: str, num_samples=5):
    print("Generating prediction samples visualization...")
    num_samples = min(num_samples, len(evaluations_data))
    if num_samples == 0: return
        
    sample_indices = np.random.choice(len(evaluations_data), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples), squeeze=False)
    fig.suptitle(f'Sample Predictions - {model_run.model_name} (Run ID: {model_run.id})', fontsize=16, fontweight='bold')
    
    for idx, sample_idx in enumerate(sample_indices):
        eval_data = evaluations_data[sample_idx]
        ax = axes[idx, 0]
        
        predictions = np.array([p[0] for p in eval_data['prediction']]).flatten()
        raw_gt = eval_data['ground_truth']
        gt_list = next(iter(raw_gt.values())) if isinstance(raw_gt, dict) else raw_gt
        ground_truth = np.array([g[0] for g in gt_list]).flatten()
        
        x = np.arange(len(predictions))
        ax.plot(x, ground_truth, 'o-', label='Ground Truth', color='green', linewidth=2)
        ax.plot(x, predictions, 's--', label='Prediction', color='red', linewidth=2)
        ax.set_title(f'Sample {idx + 1} - MSE: {eval_data["metrics"]["mse"]:.4f}, Score: {eval_data["score"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"✅ Prediction samples saved to: {save_path}")
    plt.close()

def save_inflection_plot(model, window: WindowGenerator, save_path: str, plot_col='T (degC)', max_subplots=3):
    print("Generating inflection point visualization...")
    try:
        inputs, labels = next(iter(window.test))
        predictions = model(inputs)

        plt.figure(figsize=(12, max_subplots * 4))
        plot_col_index = window.column_indices.get(plot_col)
        if plot_col_index is None: return

        for i in range(max_subplots):
            ax = plt.subplot(max_subplots, 1, i + 1)
            ax.set_ylabel(f'{plot_col} [normed]')
            ax.plot(window.input_indices, inputs[i, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10, color='blue')

            if window.label_columns:
                label_col_index = window.label_columns_indices.get(plot_col)
                if label_col_index is not None:
                    ax.scatter(window.label_indices, labels[i, :, label_col_index],
                               edgecolors='k', label='Labels', c='green', s=64)
            
            # --- FIX: Align prediction X-axis with its output length ---
            prediction_y_values = predictions[i, :, 0]
            # The x-coordinates for predictions are the LAST indices of the label window, matching the prediction length
            prediction_x_indices = window.label_indices[-len(prediction_y_values):]

            ax.scatter(prediction_x_indices, prediction_y_values,
                        marker='X', edgecolors='k', label='Predictions', c='red', s=64)
            if i == 0: ax.legend() 
        
        plt.xlabel("Time [h]")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Inflection plot saved to: {save_path}")
    except Exception as e:
        print(f"❌ Failed to generate inflection plot: {e}")
    finally:
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
        print(f"❌ Error: ModelRun with ID {args.run_id} not found."); return

    base_filename = generate_report_filename_base(model_run)
    os.makedirs("reports", exist_ok=True)
    
    chart_path = f"reports/{base_filename}_performance-charts.png"
    samples_path = f"reports/{base_filename}_prediction-samples.png"
    inflection_plot_path = f"reports/{base_filename}_inflection-analysis.png"
    csv_path = f"reports/{base_filename}_summary-stats.csv"
    
    results = db.query(Evaluation, Response, TestPrompt).join(Response, Evaluation.response_id == Response.id).join(TestPrompt, Response.test_case_id == TestPrompt.id).filter(Response.run_id == args.run_id).all()
    if not results:
        print(f"❌ Error: No evaluation results found for ModelRun ID: {args.run_id}."); return

    evaluations_data = [{'test_case_id': tc.id, 'prediction': resp.output_data.get('prediction', []), 'ground_truth': tc.ground_truth, 'metrics': ev.metrics or {}, 'score': ev.score or 0, 'passed': ev.passed} for ev, resp, tc in results]
    
    # Text report, etc. (omitted for brevity)
    
    print(f"\n  📊 Generating visualizations...")
    training_history = load_training_history(model_run.model_name)
    save_performance_charts(evaluations_data, model_run, chart_path, training_history)
    save_prediction_samples(evaluations_data, model_run, samples_path, num_samples=args.samples)
    
    # --- Inflection Plot Generation ---
    PROCESSED_DATA_PATH = "models/processed_data.csv"
    model_path = os.path.join("models", f"{model_run.model_name}.keras")
    if os.path.exists(PROCESSED_DATA_PATH) and os.path.exists(model_path):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        n = len(df)
        train_df, val_df, test_df = df[0:int(n*0.7)], df[int(n*0.7):int(n*0.9)], df[int(n*0.9):]
        try:
            custom_objects = {}
            if "baseline" in model_run.model_name:
                from ml_eval.core.implementations.train_baseline_time_series import Baseline
                custom_objects['Baseline'] = Baseline
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            # For this plot, we always use a wide window for visualization context
            plot_window = WindowGenerator(
                input_width=24, label_width=24, shift=1,
                train_df=train_df, val_df=val_df, test_df=test_df,
                label_columns=['T (degC)']
            )
            save_inflection_plot(model, plot_window, save_path=inflection_plot_path)
        except Exception as e:
            print(f"❌ Could not generate inflection plot: {e}")
    else:
        print("⚠️  Skipping inflection plot: Model file or processed data not found.")

    # ... save summary csv ...
    print("\n" + "="*70 + "\n  Report Generation Complete\n" + "="*70)

if __name__ == "__main__":
    main()