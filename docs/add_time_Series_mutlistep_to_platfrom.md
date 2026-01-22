# adding a multistep Dense model to the framework.

Ok now adding a mutltistep dense model to the framework is the same procedure as adding a dense model to the framework.  This is the perfect next step because, as you've noted, the model architecture is evolving.  And its the next step in the tf tutorial. It's no longer just processing a single time step
  but is now looking at a window of input_width=3. This is a great test for our protocol.

  The model now handles its input differently, specifically by adding a tf.keras.layers.Flatten layer to process the wider input. However, the final output shape is still (32, 1, 1), which means our existing reporting scripts will work perfectly.

  Let's walk through our established algorithm for adding this new model.

  The Plan:

   1. Stage 1: Train & Standardize: Create train_multistep_dense.py to train this new model, save it, and generate its golden dataset based on the 3-hour
      input window.
   2. Stage 2: Seed the Database: Create seed_multistep_dense.py to load the new test cases, tagging them with a new time_series_multistep_dense type.
   3. Stage 3: Integrate & Evaluate: Add a new elif block to run_evaluation.py to recognize the new model type.



## train & standardize.  
lets again look at the differences between the dense and mulit-step dense model.  
So lets use the compare function in vs code and see what we find.  Window_genereator and create_golden_dataset are exactly the same.  In main(), the windowGeneraor instance is manufactured a little differently.  And the flatten function is added to the creation/compilation of the model. Other than that its mostly metadata, ie dense vs multistep_dense, differetn path names etc.  This is geting to be old hat, when the shapes and inputs and data is all the same.  (ooooh i thnk that is what they call foreshadowing!)  Ok thats pretty straightforward.  Also dirve home the point that windowgenerator , and helper functions should be in  their own folder.  This copy n' pasting is a recipe for disaster. 

## Seed the database.  
We see the same pattern here that we saw above and when we compare the desne with linear.  The shape of the model is the same, and the inputs vary slightly, so again all we are channging/amending here is the metadata.  This again highlights a target for refactoring.  See the metadata.yml aside in the dense vs linear wrtieup.


## Integrate and evaluate.  
Now add the final elif block to handle our new "time_series_multistep_dense" model. Even though this model uses a wider input window, our generic KerasTimeSeriesModel and KerasTimeSeriesAdapter are robust enough to handle it.  This is simple. For the reporting, the current report generator was fine, but i thought we should add a chart looking at the timeline of the predictions. We know the model has trouble around inflection points (daily high/low temp).  So lets get a chart that shows that.


## Execution 
Did the plan work?  Let me run thru the commands. 
```
python scripts/train_multistep_dense.py --num_samples 50
Loaded processed data: (70091, 19)
Train: 49063, Val: 14018, Test: 7010

New Window Configuration:
Traceback (most recent call last):
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/scripts/train_multistep_dense.py", line 216, in <module>
    main(args)
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/scripts/train_multistep_dense.py", line 151, in main
    print(f"  Input shape: {conv_window.example[0].shape}")
                            ^^^^^^^^^^^^^^^^^^^
AttributeError: 'WindowGenerator' object has no attribute 'example'
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ python scripts/train_multistep_dense.py --num_samples 50
Loaded processed data: (70091, 19)
Train: 49063, Val: 14018, Test: 7010
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1768866074.733322 3503324 gpu_device.cc:2342] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...

Training Multi-step Dense model...
Epoch 1/20
1534/1534 - 2s - 1ms/step - loss: 0.0245 - mean_absolute_error: 0.0977 - val_loss: 0.0129 - val_mean_absolute_error: 0.0886
Epoch 2/20
1534/1534 - 2s - 989us/step - loss: 0.0087 - mean_absolute_error: 0.0678 - val_loss: 0.0104 - val_mean_absolute_error: 0.0771
Epoch 3/20
1534/1534 - 2s - 1ms/step - loss: 0.0079 - mean_absolute_error: 0.0640 - val_loss: 0.0101 - val_mean_absolute_error: 0.0744
Epoch 4/20
1534/1534 - 2s - 1ms/step - loss: 0.0075 - mean_absolute_error: 0.0621 - val_loss: 0.0070 - val_mean_absolute_error: 0.0605
Epoch 5/20
1534/1534 - 2s - 1ms/step - loss: 0.0072 - mean_absolute_error: 0.0605 - val_loss: 0.0099 - val_mean_absolute_error: 0.0700
Epoch 6/20
1534/1534 - 2s - 1ms/step - loss: 0.0071 - mean_absolute_error: 0.0598 - val_loss: 0.0072 - val_mean_absolute_error: 0.0621
✅ Multi-step Dense model trained!

Saving Multi-step Dense model to models/multistep_dense_model.keras...
✅ Multi-step Dense model saved!

============================================================
Creating Golden Dataset for Evaluation
============================================================
Creating a sample of 50 test cases.

✅ Created 50 golden test cases
   Saved to: data/multistep_dense_golden_dataset.json
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ python scripts/seed_multistep_dense.py
--- Seeding Multi-step Dense Test Cases from Golden Dataset ---
Loaded 50 test cases from golden dataset

=============================================================
Seeding Complete!
  ✅ Handled: 50/50 test cases
============================================================

Next steps:
  1. Create a model run:
     curl -X 'POST' 'http://localhost:8000/api/v1/runs/'        -H 'Content-Type: application/json'        -d '{"model_name": "multistep_dense_model", "model_version": "1.0", "model_type": "time_series_multistep_dense"}'

  2. Run evaluation:
     python scripts/run_evaluation.py <RUN_ID>

  3. Generate report:
     python scripts/gemerate_report_time_series_v3.py <RUN_ID>
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'POST' 'http://localhost:8000/api/v1/runs/' -H 'Content-Type: application/json' -d '{"model_name": "multistep_dense_model", "model_version": "1.0", "model_type": "time_series_multistep_dense"}'
{"model_name":"multistep_dense_model","model_version":"1.0","model_type":"time_series_multistep_dense","model_endpoint":null,"config":{},"id":5,"status":"pending","started_at":"2026-01-19T23:43:07.314440","completed_at":null,"total_cases":0,"completed_cases":0,"failed_cases":0}(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ python scripts/run_evaluation.py 5
2026-01-19 15:43:26.650595: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-01-19 15:43:26.681446: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-01-19 15:43:27.647375: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
--- Setting up evaluation for ModelRun ID: 5 ---
--- Initializing components ---
--- Loading trained Keras time series model from: models/multistep_dense_model.keras ---
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1768866210.610490 3505554 gpu_device.cc:2342] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
✅ Model loaded successfully.
✅ Components initialized for model_type: time_series_multistep_dense.
--- Instantiating Evaluation Engine ---
✅ Engine instantiated.
--- Running evaluation for ModelRun ID: 5 ---
Starting evaluation for ModelRun 5 (multistep_dense_model 1.0)...
Evaluation for ModelRun 5 completed.

🎉 Evaluation complete for ModelRun ID: 5
   - Total Cases: 50
   - Completed: 50
   - Failed: 0
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ python scripts/generate_report_time_series_v3.py 5
2026-01-19 15:43:52.906576: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-01-19 15:43:52.938952: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-01-19 15:43:53.807182: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
--- Generating Time Series Report for ModelRun ID: 5 ---
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1768866236.343661 3508331 gpu_device.cc:2342] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
Generating inflection point visualization...
❌ Failed to generate inflection plot: Sequential model 'sequential' has already been configured to use input shape (None, 3, 19). You cannot build it with input_shape (32, 24, 19)
  Error details: Sequential model 'sequential' has already been configured to use input shape (None, 3, 19). You cannot build it with input_shape (32, 24, 19)

======================================================================
  TIME SERIES PERFORMANCE REPORT
  Model: multistep_dense_model (v1.0)
  Run ID: 5
  Model Type: time_series_multistep_dense
======================================================================

  📊 Overall Statistics:
     Total Test Cases: 50
     Passed: 45 (90.00%)
     Failed: 5 (10.00%)

  📈 Error Metrics:
     Mean Squared Error (MSE):
        Mean:   0.005365
        Median: 0.002544
        Std:    0.007070
        Min:    0.000000
        Max:    0.032957

     Mean Absolute Error (MAE):
        Mean:   0.058206
        Median: 0.050400
        Std:    0.044466
        Min:    0.000584
        Max:    0.181541

     Root Mean Squared Error (RMSE):
        Mean:   0.058206
        Median: 0.050400
        Std:    0.044466
        Min:    0.000584
        Max:    0.181541

  🎯 Normalized Score (1.0 = Perfect):
     Mean:   0.883021
        Median: 0.996557
        Std:    0.285150
        Min:    0.000000
     Max:    1.000000

  🏆 Best Predictions (Lowest MSE):
     1. Test Case 205: MSE=0.000000, Score=1.0000
     2. Test Case 249: MSE=0.000003, Score=0.9962
     3. Test Case 237: MSE=0.000008, Score=0.9999

  ⚠️  Worst Predictions (Highest MSE):
     1. Test Case 216: MSE=0.018693, Score=0.9498
     2. Test Case 230: MSE=0.025704, Score=0.9943
     3. Test Case 202: MSE=0.032957, Score=0.8544

  📊 Generating visualizations...
Traceback (most recent call last):
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/scripts/generate_report_time_series_v3.py", line 439, in <module>
    main()
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/scripts/generate_report_time_series_v3.py", line 412, in main
    training_history = load_training_history(model_run.model_name)
                       ^^^^^^^^^^^^^^^^^^^^^
NameError: name 'load_training_history' is not defined. Did you mean: 'training_history'?
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ python scripts/generate_report_time_series_v3.py 5
2026-01-19 15:48:41.187829: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-01-19 15:48:41.220328: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-01-19 15:48:42.056883: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
--- Generating Time Series Report for ModelRun ID: 5 ---
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1768866524.502506 3511716 gpu_device.cc:2342] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
Model expects input width of 3. Creating matching window for plot.
Generating inflection point visualization...
✅ Inflection plot saved to: reports/multistep-dense-model_v1.0_run-5_inflection-analysis.png
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ python scripts/generate_report_time_series_v3.py 5
2026-01-19 15:53:47.192597: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-01-19 15:53:47.223458: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-01-19 15:53:48.072687: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
--- Generating Time Series Report for ModelRun ID: 5 ---

  📊 Generating visualizations...
Generating performance charts...
✅ Performance charts saved to: reports/multistep-dense-model_v1.0_run-5_performance-charts.png
Generating prediction samples visualization...
✅ Prediction samples saved to: reports/multistep-dense-model_v1.0_run-5_prediction-samples.png
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1768866831.636994 3515264 gpu_device.cc:2342] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...

Model expects input width of 3. Creating matching window for inflection plot.
Generating inflection point visualization...
✅ Inflection plot saved to: reports/multistep-dense-model_v1.0_run-5_inflection-analysis.png
  ✅ Summary CSV saved to: reports/multistep-dense-model_v1.0_run-5_summary-stats.csv

======================================================================
  Report Generation Complete
======================================================================
```


There were some hiccups, and the inflection graph is not what i intended, but I can fix that in post.  Right. Hmmm. there is some discrepancy in how the reports are shown, too. Blargh.