# scripts/train_rnn_time_series.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt

MODEL_DIR = "models"
REPORTS_DIR = "reports"
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
            sequence_stride=1, shuffle=True, batch_size=32)
        return ds.map(self.split_window)

    @property
    def train(self): return self.make_dataset(self.train_df)
    @property
    def val(self): return self.make_dataset(self.val_df)
    @property
    def test(self): return self.make_dataset(self.test_df)

def create_golden_dataset(test_df, window, num_samples=50, output_path='data/golden_dataset.json'):
    input_width = window.input_width
    label_width = window.label_width
    shift = window.shift
    label_column = window.label_columns[0]
    
    golden_cases = []
    max_start_idx = len(test_df) - (input_width + shift)
    
    if num_samples == -1 or num_samples > max_start_idx + 1:
        print(f"Using all {max_start_idx + 1} possible test cases for golden dataset.")
        indices = np.arange(max_start_idx + 1)
    else:
        print(f"Creating a sample of {num_samples} test cases for golden dataset.")
        indices = np.linspace(0, max_start_idx, num_samples, dtype=int)

    for i, start_idx in enumerate(indices):
        input_end = start_idx + input_width
        input_window = test_df.iloc[start_idx:input_end].values.tolist()
        
        label_start = start_idx + input_width + shift - 1
        label_end = label_start + label_width
        ground_truth = test_df.iloc[label_start:label_end][[label_column]].values.tolist()
        
        golden_case = {
            "case_id": i + 1, "input_data": {"window": input_window},
            "ground_truth": {"prediction": ground_truth},
            "metadata": {"start_index": int(start_idx), "input_width": input_width, "label_width": label_width, "shift": shift, "label_column": label_column}
        }
        golden_cases.append(golden_case)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f: json.dump(golden_cases, f, indent=2)
    
    print(f"\n✅ Created {len(golden_cases)} golden test cases and saved to: {output_path}")
    return golden_cases

def plot_inflection_predictions(model, window, plot_col='T (degC)', max_subplots=3, save_path=None):
    inputs, labels = next(iter(window.test))
    predictions = model(inputs)

    plt.figure(figsize=(12, max_subplots * 4))
    plot_col_index = window.column_indices[plot_col]
    
    for i in range(max_subplots):
        ax = plt.subplot(max_subplots, 1, i + 1)
        ax.set_ylabel(f'{plot_col} [normed]')
        
        ax.plot(window.input_indices, inputs[i, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10, color='blue')

        if window.label_columns:
            label_col_index = window.label_columns_indices.get(plot_col)
            if label_col_index is not None:
                # For RNN with wide window but single output, labels will have many points
                ax.scatter(window.label_indices, labels[i, :, label_col_index],
                           edgecolors='k', label='Labels', c='green', s=64)

        # For single-output RNN, predictions shape is (batch, 1)
        # We need to plot this single point at the correct time step
        prediction_x_index = window.label_indices[0] # For shift=1, this is correct
        ax.scatter([prediction_x_index], [predictions.numpy()[i,0]],
                    marker='X', edgecolors='k', label='Predictions', c='red', s=64)

        if i == 0: ax.legend() 
    
    plt.xlabel("Time [h]")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Inflection point analysis chart saved to: {save_path}")
    else:
        plt.show()

def main(args):
    """Main function to train the RNN (LSTM) model and generate datasets."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ Processed data not found at {PROCESSED_DATA_PATH}.")
        exit(1)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loaded processed data: {df.shape}")

    n = len(df)
    train_df, val_df, test_df = df[0:int(n*0.7)], df[int(n*0.7):int(n*0.9)], df[int(n*0.9):]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    INPUT_WIDTH = 24
    # This window is used for both training and plotting
    rnn_window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=1,
        shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['T (degC)']
    )

    print("\nRNN Window Configuration:")
    print(f"  Input shape: {rnn_window.example[0].shape}")
    print(f"  Label shape: {rnn_window.example[1].shape}")

    rnn_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(units=1)
    ])

    rnn_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    MAX_EPOCHS = 20
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min', restore_best_weights=True)

    print("\nTraining RNN model...")
    rnn_model.fit(rnn_window.train, epochs=MAX_EPOCHS,
                  validation_data=rnn_window.val,
                  callbacks=[early_stopping],
                  verbose=2)
    print("✅ RNN model trained!")

    model_path = os.path.join(MODEL_DIR, "rnn_model.keras")
    print(f"\nSaving RNN model to {model_path}...")
    rnn_model.save(model_path)
    print("✅ RNN model saved!")

    print("\n" + "="*60 + "\nCreating Golden Dataset for Evaluation\n" + "="*60)
    create_golden_dataset(test_df, rnn_window, num_samples=args.num_samples,
                          output_path='data/rnn_golden_dataset.json')

    print("\n" + "="*60 + "\nGenerating Inflection Point Visualization\n" + "="*60)
    plot_save_path = os.path.join(REPORTS_DIR, "rnn_inflection_plot.png")
    plot_inflection_predictions(rnn_model, rnn_window, save_path=plot_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RNN Time Series model.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples for the golden dataset.")
    args = parser.parse_args()
    main(args)


#     Here is the final command sequence to run the end-to-end evaluation for the RNN model:

#   Step 1: Train the RNN Model

#   This uses the wide 24-hour input window and will create rnn_model.keras and its corresponding assets.

#    1 python scripts/train_rnn_time_series.py --num_samples 50

#   Step 2: Seed the Database

#   Load the test cases into the database with the "time_series_rnn" type.

#    1 python scripts/seed_rnn_test_cases.py

#   Step 3: Create the Evaluation Run

#   Register the run for the RNN model.

#    1 curl -X 'POST' 'http://localhost:8000/api/v1/runs/' \
#    2   -H 'Content-Type: application/json' \
#    3   -d '{"model_name": "rnn_model", "model_version": "1.0", "model_type": "time_series_rnn"}'
#   Use the run_id returned by this command in the next step.

#   Step 4: Run the Evaluation

#   Execute the evaluation engine with your new <RUN_ID>.

#    1 python scripts/run_evaluation.py <YOUR_RUN_ID>

#   Step 5: Generate the Report

#   Finally, generate the full report, including the inflection analysis, using our v3 script.

#    1 python scripts/generate_report_time_series_v3.py <YOUR_RUN_ID>
