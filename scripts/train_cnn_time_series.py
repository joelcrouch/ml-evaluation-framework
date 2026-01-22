# scripts/train_cnn_time_series.py
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

def create_golden_dataset(test_df, window, num_samples=50, output_path='data/golden_dataset.json'):
    input_width = window.input_width
    label_width = window.label_width
    shift = window.shift
    label_column = window.label_columns[0] # Assume one label column
    
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
    """Generates and saves a plot of model predictions over a wide time window."""
    inputs, labels = next(iter(window.test))
    predictions = model(inputs)

    plt.figure(figsize=(12, max_subplots * 4))
    plot_col_index = window.column_indices[plot_col]
    
    for i in range(max_subplots):
        ax = plt.subplot(max_subplots, 1, i + 1)
        ax.set_ylabel(f'{plot_col} [normed]')
        
        # Plot inputs
        ax.plot(window.input_indices, inputs[i, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10, color='blue')

        # Plot labels (ground truth)
        if window.label_columns:
            label_col_index = window.label_columns_indices.get(plot_col, None)
            if label_col_index is not None:
                ax.scatter(window.label_indices, labels[i, :, label_col_index],
                           edgecolors='k', label='Labels', c='green', s=64)

        # Plot predictions
        ax.scatter(window.label_indices, predictions[i, :, 0],
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
    """Main function to train the CNN model and generate datasets."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ Processed data not found at {PROCESSED_DATA_PATH}.")
        exit(1)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loaded processed data: {df.shape}")

    n = len(df)
    train_df, val_df, test_df = df[0:int(n*0.7)], df[int(n*0.7):int(n*0.9)], df[int(n*0.9):]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # --- Window for training the model (3-hour input, same as multi-step dense) ---
    CONV_WIDTH = 3
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['T (degC)']
    )

    # print("\nNew Window Configuration (same as Multi-step Dense):")
    # print(f"  Input shape: {conv_window.example[0].shape}")
    # print(f"  Label shape: {conv_window.example[1].shape}")


    # --- Create and compile the CNN model ---
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH,),
                               activation='relu',
                               input_shape=(conv_window.input_width, len(df.columns))),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    cnn_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    # --- Train the CNN model ---
    MAX_EPOCHS = 20
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=2,
                                                       mode='min',
                                                       restore_best_weights=True)

    print("\nTraining CNN model...")
    history = cnn_model.fit(conv_window.train, epochs=MAX_EPOCHS,
                              validation_data=conv_window.val,
                              callbacks=[early_stopping],
                              verbose=2)
    print("✅ CNN model trained!")

    model_path = os.path.join(MODEL_DIR, "cnn_model.keras")
    print(f"\nSaving CNN model to {model_path}...")
    cnn_model.save(model_path)
    print("✅ CNN model saved!")

    # --- Create golden dataset ---
    print("\n" + "="*60 + "\nCreating Golden Dataset for Evaluation\n" + "="*60)
    create_golden_dataset(test_df, conv_window, num_samples=args.num_samples,
                          output_path='data/cnn_golden_dataset.json')

    # --- Generate inflection point visualization ---
    # print("\n" + "="*60 + "\nGenerating Inflection Point Visualization\n" + "="*60)
    # wide_window = WindowGenerator(
    #     input_width=24,
    #     label_width=24,
    #     shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     label_columns=['T (degC)']
    # )
    # plot_save_path = os.path.join(REPORTS_DIR, "cnn_inflection_plot.png")
    # plot_inflection_predictions(cnn_model, wide_window, save_path=plot_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN Time Series model.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples for the golden dataset. Use -1 for all possible samples."
    )
    args = parser.parse_args()
    main(args)
