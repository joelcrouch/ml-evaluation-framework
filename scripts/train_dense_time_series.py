# train_dense_time_series.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import argparse

MODEL_DIR = "models"
PROCESSED_DATA_PATH = "models/processed_data.csv" 

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

def create_golden_dataset(test_df, window_config, num_samples=50, output_path='data/golden_dataset.json'):
    """Creates a golden dataset from the test set for evaluation."""
    input_width = window_config['input_width']
    label_width = window_config['label_width']
    shift = window_config['shift']
    label_column = window_config['label_column']
    
    golden_cases = []
    
    max_start_idx = len(test_df) - (input_width + shift)
    
    if num_samples == -1 or num_samples > max_start_idx + 1:
        print(f"Using all {max_start_idx + 1} possible test cases.")
        indices = np.arange(max_start_idx + 1)
    else:
        print(f"Creating a sample of {num_samples} test cases.")
        indices = np.linspace(0, max_start_idx, num_samples, dtype=int)

    for i, start_idx in enumerate(indices):
        input_end = start_idx + input_width
        input_window = test_df.iloc[start_idx:input_end].values.tolist()
        
        label_start = start_idx + input_width + shift - 1
        label_end = label_start + label_width
        ground_truth = test_df.iloc[label_start:label_end][[label_column]].values.tolist()
        
        golden_case = {
            "case_id": i + 1,
            "input_data": {"window": input_window},
            "ground_truth": {"prediction": ground_truth},
            "metadata": {
                "start_index": int(start_idx),
                "input_width": input_width,
                "label_width": label_width,
                "shift": shift,
                "label_column": label_column
            }
        }
        golden_cases.append(golden_case)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(golden_cases, f, indent=2)
    
    print(f"\n✅ Created {len(golden_cases)} golden test cases")
    print(f"   Saved to: {output_path}")
    
    return golden_cases

def main(args):
    """Main function to train the Dense model and generate datasets."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ Processed data not found at {PROCESSED_DATA_PATH}.")
        exit(1)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loaded processed data: {df.shape}")

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    single_step_window = WindowGenerator(
        input_width=1, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['T (degC)']
    )

    # --- Create and compile the Dense model ---
    dense_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    dense_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    # --- Train the Dense model ---
    MAX_EPOCHS = 20
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=2,
                                                       mode='min',
                                                       restore_best_weights=True)

    print("\nTraining Dense model...")
    history = dense_model.fit(single_step_window.train, epochs=MAX_EPOCHS,
                               validation_data=single_step_window.val,
                               callbacks=[early_stopping],
                               verbose=2)
    print("✅ Dense model trained!")

    model_path = os.path.join(MODEL_DIR, "dense_model.keras")
    print(f"\nSaving Dense model to {model_path}...")
    dense_model.save(model_path)
    print("✅ Dense model saved!")

    # --- Create golden dataset ---
    print("\n" + "="*60)
    print("Creating Golden Dataset for Evaluation")
    print("="*60)

    create_golden_dataset(
        test_df=test_df,
        window_config={
            'input_width': single_step_window.input_width,
            'label_width': single_step_window.label_width,
            'shift': single_step_window.shift,
            'label_column': 'T (degC)'
        },
        num_samples=args.num_samples,
        output_path='data/dense_golden_dataset.json'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Dense Time Series model and create a golden dataset.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples for the golden dataset. Use -1 for all possible samples."
    )
    args = parser.parse_args()
    main(args)
