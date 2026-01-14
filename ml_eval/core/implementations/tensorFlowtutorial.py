import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_DIR = "/home/dell-linux-dev3/Projects/ml-evaluation-framework/models"
PROCESSED_DATA_PATH = "/home/dell-linux-dev3/Projects/ml-evaluation-framework/models/processed_data.csv"
MAX_EPOCHS = 20

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # 1. Store the raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # 2. Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # 3. Work out the window parameters
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
        """Splits the window into inputs and labels."""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes manually.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """Converts a pandas dataframe into a tensorflow dataset."""
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

    # Properties to access the datasets easily
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch for plotting."""
        if not hasattr(self, '_example'):
            self._example = next(iter(self.train))
        return self._example


    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')

            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label='Inputs',
                marker='.',
                zorder=-10
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors='k',
                label='Labels',
                s=64
            )

            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker='X',
                    edgecolors='k',
                    label='Predictions',
                    s=64
                )

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index=label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result=inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

def compile_and_fit(model, window, model_name, patience=2):
    checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}.keras")

    # 1. Check if model already exists
    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] Model '{model_name}' found at {checkpoint_path}. Loading weights...")
        # We build the model by calling it on an example batch so shapes are initialized
        model(window.example[0]) 
        model.load_weights(checkpoint_path)
        
        # Re-compile so the model is ready for evaluation
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
        return None # No history to return if we didn't train

    # 2. If it doesn't exist, proceed with training
    print(f"\n[INFO] Model '{model_name}' not found. Starting training...")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, mode='min')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping, checkpoint_callback])
    return history

def get_data(csv_path, force_process=False):
    if os.path.exists(PROCESSED_DATA_PATH) and not force_process:
        print(f"\n[INFO] Loading processed data from: {PROCESSED_DATA_PATH}")
        df = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        print("\n[INFO] Processing raw data and engineering features...")
        df = pd.read_csv(csv_path)
        df = df[5::6] # Hourly subsampling
        
        # 1. Date/Time processing
        date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
        timestamp_s = date_time.map(pd.Timestamp.timestamp)

        # 2. Cleaning Outliers
        df.loc[df['wv (m/s)'] == -9999.0, 'wv (m/s)'] = 0.0
        df.loc[df['max. wv (m/s)'] == -9999.0, 'max. wv (m/s)'] = 0.0

        # 3. Wind Vector Engineering
        wv = df.pop('wv (m/s)')
        max_wv = df.pop('max. wv (m/s)')
        wd_rad = df.pop('wd (deg)') * np.pi / 180
        df['Wx'] = wv * np.cos(wd_rad)
        df['Wy'] = wv * np.sin(wd_rad)
        df['max Wx'] = max_wv * np.cos(wd_rad)
        df['max Wy'] = max_wv * np.sin(wd_rad)

        # 4. Time of Day/Year Signals (Periodic)
        day = 24*60*60
        year = (365.2425)*day
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

        # Create the column indices dictionary before returning
    column_indices = {name: i for i, name in enumerate(df.columns)}
    # 5. Normalization (using Train stats only)
    n = len(df)
    train_df_raw = df[0:int(n*0.7)]
    train_mean = train_df_raw.mean()
    train_std = train_df_raw.std()
    
    df = (df - train_mean) / train_std
    
    # Cache to disk
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[SUCCESS] Processed data cached at {PROCESSED_DATA_PATH}")

    # Final Splits
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]
    
    # Save the test data to a new file
    # /home/dell-linux-dev3/Projects/ml-evaluation-framework/data
    test_data_path = '/home/dell-linux-dev3/Projects/ml-evaluation-framework/data/weather_test_data.csv'
    test_df.to_csv(test_data_path, index=False)
    print(f"[SUCCESS] Golden test dataset saved to {test_data_path}")


    # Return the dataframes AND the indices
    return train_df, val_df, test_df, column_indices


# 1. Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Cleaner terminal output
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# 2. Locate the file (already downloaded)
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

base_dir = os.path.dirname(zip_path)
csv_files = glob.glob(f"{base_dir}/**/*.csv", recursive=True)
# csv_path = csv_files[0]
csv_path = glob.glob(f"{base_dir}/**/*.csv", recursive=True)[0]
# print(f"Found CSV at: {csv_path}")

# 2. Get Data (This is the "Skip" logic)
train_df, val_df, test_df, column_indices = get_data(csv_path)

w1 = WindowGenerator(

    input_width=24,
    label_width=1,
    shift=24,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=['T (degC)']
)

print(w1)
example_inputs, example_labels = next(iter(w1.train))
print("******")
print(example_inputs.shape)
print(example_labels.shape)
print("********")

print("**************")
w2 = WindowGenerator(
    input_width=6,
    label_width=1,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=['T (degC)']
)

print(w2)
print("********")

example_window = tf.stack([
    np.array(train_df[:w2.total_window_size]),
    np.array(train_df[100:100+w2.total_window_size]),
    np.array(train_df[200:200+w2.total_window_size])
])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')


print("PLOT")
w2._example = (example_inputs, example_labels)
w2.plot(plot_col='p (mbar)')
plt.show()
w2.plot()
plt.show()

# Each element is an (inputs, label) pair.
w2.train.element_spec
for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

print("2nd way")
example_inputs, example_labels = next(iter(w2.train))

print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
print(f'Labels shape (batch, time, features): {example_labels.shape}')


single_step_window = WindowGenerator(
    input_width=1,
    label_width=1,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=['T (degC)']
)
print("single step")
print(single_step_window)


for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')


baseline = Baseline(label_index=column_indices['T (degC)'])

baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val, return_dict=True)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)

print("wide window")
wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=['T (degC)'])

print(wide_window)
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)
wide_window.plot(baseline)
plt.show()

print("linear model")
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)


# history = compile_and_fit(linear, single_step_window)
history = compile_and_fit(linear, single_step_window, model_name="linear_model")

val_performance['Linear'] = linear.evaluate(single_step_window.val, return_dict=True)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0, return_dict=True)


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', linear(wide_window.example[0]).shape)
wide_window.plot(linear)
plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
plt.show()

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
# history = compile_and_fit(dense, single_step_window)
history = compile_and_fit(dense, single_step_window, model_name="dense_model")

val_performance['Dense'] = dense.evaluate(single_step_window.val, return_dict=True)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', dense(wide_window.example[0]).shape)

# Plot predictions vs labels
wide_window.plot(dense)

plt.figure(figsize=(12, 6))
plt.bar(x = range(len(train_df.columns)),
        height=dense.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
plt.title("Dense layer 1 weights")

plt.show()
plt.figure(figsize=(12, 6))
plt.bar(x = range(len(train_df.columns)),
        height=dense.layers[0].kernel[:,1].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
plt.title("Dense layer 2 weights")

# Compare test MAE for all models
plt.figure(figsize=(8,5))
plt.bar(performance.keys(), [v['mean_absolute_error'] for v in performance.values()])
plt.ylabel('Mean Absolute Error (Test)')
plt.title('Model Performance Comparison')
plt.show()

CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=['T (degC)'])

print(conv_window)

conv_window.plot()
plt.suptitle("Given 3 hours of inputs, predict 1 hour into the future.")
plt.show()

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)
history = compile_and_fit(multi_step_dense, conv_window, model_name="multi_step_dense")
# history = compile_and_fit(multi_step_dense, conv_window)

# IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val, return_dict=True)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0, return_dict=True)
conv_window.plot(multi_step_dense)
plt.show()


print("*****************************************")
print("CNN")
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])
print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)
history = compile_and_fit(conv_model, conv_window, model_name="cnn")
val_performance['Conv'] = conv_model.evaluate(conv_window.val, return_dict=True)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0, return_dict=True)
print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=['T (degC)'])

print(wide_conv_window)

print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)
wide_conv_window.plot(conv_model)
plt.show()