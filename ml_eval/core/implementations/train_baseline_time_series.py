# train_baseline.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
import glob

MODEL_DIR = "models"
PROCESSED_DATA_PATH = "models/processed_data.csv"

@tf.keras.utils.register_keras_serializable(package="TimeSeriesModels")
class Baseline(tf.keras.Model):
    def __init__(self, label_index=None, **kwargs):
        super().__init__(**kwargs)
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    
    def get_config(self):
        config = super().get_config()
        config.update({'label_index': self.label_index})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

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
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None, **kwargs):
        super().__init__(**kwargs)
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    
    def get_config(self):
        config = super().get_config()
        config.update({'label_index': self.label_index})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load processed data (assumes you already have this from tutorial)
if not os.path.exists(PROCESSED_DATA_PATH):
    print("❌ Processed data not found. Run the full tutorial script first to generate it.")
    exit(1)

df = pd.read_csv(PROCESSED_DATA_PATH)
print(f"Loaded processed data: {df.shape}")

# Split data (70% train, 20% val, 10% test)
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

column_indices = {name: i for i, name in enumerate(df.columns)}

# Create single-step window (matching tutorial)
single_step_window = WindowGenerator(
    input_width=1,
    label_width=1,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=['T (degC)']
)

print(f"\nWindow configuration:")
print(f"  Input width: {single_step_window.input_width}")
print(f"  Label width: {single_step_window.label_width}")
print(f"  Shift: {single_step_window.shift}")

# Create baseline model
baseline = Baseline(label_index=column_indices['T (degC)'])

# Compile
baseline.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

print("\nEvaluating baseline on validation set...")
val_performance = baseline.evaluate(single_step_window.val, return_dict=True)
print(f"Validation - Loss: {val_performance['loss']:.4f}, MAE: {val_performance['mean_absolute_error']:.4f}")

print("\nEvaluating baseline on test set...")
test_performance = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)
print(f"Test - Loss: {test_performance['loss']:.4f}, MAE: {test_performance['mean_absolute_error']:.4f}")

# Save model
model_path = os.path.join(MODEL_DIR, "baseline_model.keras")
print(f"\nSaving baseline model to {model_path}...")
baseline.save(model_path)
print("✅ Baseline model saved!")

# Test loading
# print("\nTesting model loading...")
# loaded_baseline = tf.keras.models.load_model(model_path)
# print("✅ Model loaded successfully!")

print("\nTesting model loading...")
loaded_baseline = tf.keras.models.load_model(
    model_path,
    custom_objects={'Baseline': Baseline}
)
print("✅ Model loaded successfully!")

# Verify loaded model works
test_perf_loaded = loaded_baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)
print(f"Loaded model test performance: Loss: {test_perf_loaded['loss']:.4f}, MAE: {test_perf_loaded['mean_absolute_error']:.4f}")

def create_golden_dataset(test_df, window_config, num_samples=50, output_path='data/baseline_golden_dataset.json'):
    """
    Creates a golden dataset from the test set for evaluation.
    
    Args:
        test_df: Test dataframe
        window_config: Dictionary with input_width, label_width, shift, label_column
        num_samples: Number of test cases to create
        output_path: Where to save the golden dataset
    """
    import json
    
    input_width = window_config['input_width']
    label_width = window_config['label_width']
    shift = window_config['shift']
    label_column = window_config['label_column']
    
    golden_cases = []
    
    # Calculate max start index
    max_start = len(test_df) - (input_width + shift)
    
    # Create evenly spaced samples across the test set
    indices = np.linspace(0, max_start, num_samples, dtype=int)
    
    for i, start_idx in enumerate(indices):
        # Input window: input_width timesteps, all features
        input_end = start_idx + input_width
        input_window = test_df.iloc[start_idx:input_end].values.tolist()
        
        # Ground truth: label_width timesteps, only temperature
        label_start = start_idx + input_width + shift - 1
        label_end = label_start + label_width
        ground_truth = test_df.iloc[label_start:label_end][[label_column]].values.tolist()
        
        golden_case = {
            "case_id": i + 1,
            "input_data": {
                "window": input_window
            },
            "ground_truth": {
                "prediction": ground_truth
            },
            "metadata": {
                "start_index": int(start_idx),
                "input_width": input_width,
                "label_width": label_width,
                "shift": shift,
                "label_column": label_column
            }
        }
        
        golden_cases.append(golden_case)
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(golden_cases, f, indent=2)
    
    print(f"\n✅ Created {len(golden_cases)} golden test cases")
    print(f"   Saved to: {output_path}")
    
    return golden_cases

# Create golden dataset
print("\n" + "="*60)
print("Creating Golden Dataset for Evaluation")
print("="*60)

golden_dataset = create_golden_dataset(
    test_df=test_df,
    window_config={
        'input_width': single_step_window.input_width,
        'label_width': single_step_window.label_width,
        'shift': single_step_window.shift,
        'label_column': 'T (degC)'
    },
    num_samples=50,
    output_path='data/baseline_golden_dataset.json'
)

# Show a sample
print("\nSample golden test case:")
print(f"  Input shape: {len(golden_dataset[0]['input_data']['window'])} timesteps × {len(golden_dataset[0]['input_data']['window'][0])} features")
print(f"  Ground truth shape: {len(golden_dataset[0]['ground_truth']['prediction'])} timesteps × {len(golden_dataset[0]['ground_truth']['prediction'][0])} features")
print(f"  First input timestep (first 5 features): {golden_dataset[0]['input_data']['window'][0][:5]}")
print(f"  Ground truth temperature: {golden_dataset[0]['ground_truth']['prediction']}")


print("\n" + "="*60)
print("Generating Tutorial-Style Visualizations")
print("="*60)

# Plot the baseline predictions on the validation set
print("\nCreating prediction visualization...")
import matplotlib.pyplot as plt

# Get a batch from validation set
for example_inputs, example_labels in single_step_window.val.take(1):
    example_predictions = baseline.predict(example_inputs)

# Create the tutorial-style plot
plt.figure(figsize=(12, 8))
max_n = 3

for n in range(max_n):
    plt.subplot(max_n, 1, n + 1)
    plt.ylabel('T (degC) [normed]')
    
    # Input indices (time 0)
    input_indices = [0]
    # Label indices (time 1)  
    label_indices = [1]
    
    # Plot inputs
    plt.plot(input_indices, 
             example_inputs[n, :, column_indices['T (degC)']].numpy(),
             label='Inputs', marker='.', markersize=15, zorder=-10)
    
    # Plot labels (ground truth)
    plt.scatter(label_indices,
                example_labels[n, :, 0].numpy(),
                edgecolors='k', label='Labels', c='green', s=64)
    
    # Plot predictions
    plt.scatter(label_indices,
                example_predictions[n, :, 0],
                marker='X', edgecolors='k', label='Predictions', 
                c='red', s=64)
    
    if n == 0:
        plt.legend()
    
    plt.xticks([0, 1], ['t=0 (current)', 't=1 (predicted)'])

plt.xlabel('Time [h]')
plt.tight_layout()
# plt.savefig('reports/baseline_training_predictions.png', dpi=300, bbox_inches='tight')
plt.savefig('reports/baseline_model_training_predictions.png', dpi=300, bbox_inches='tight')
print("✅ Tutorial-style chart saved to: reports/baseline_training_predictions.png")
plt.show()