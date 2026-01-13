activate the conda environment;
```
conda activate ml-eval-framework
```
run tests
```
pytest tests\
```  or something like that.

bring up the instances (db)
```./start_db.py
```
and start the fastapi 
```
uvicorn ml_eval.main:app --host 0.0.0.0 --port 8000

```

test the fastapi with something like this;
```
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs 

```
you shuld see '200' in your prompt

Sometimes you gotta do this :
```
conda env update --file environment.yml --prune 
``` to update and clean the environment. 

demo example: python scripts/run_evaluation.py 1^Cation-framework$ python scripts/run_evaluation.py 6
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ python scripts/run_evaluation.py 1
2026-01-12 23:51:29.021399: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-01-12 23:51:29.052875: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-01-12 23:51:30.126005: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
--- Setting up evaluation for ModelRun ID: 1 ---
--- Initializing components ---
--- Loading trained flower classifier model... ---
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1768290693.036654  716541 gpu_device.cc:2342] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
✅ Model loaded. Class names: ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
✅ Components initialized for model_type: image_classification.
--- Instantiating Evaluation Engine ---
✅ Engine instantiated.
--- Running evaluation for ModelRun ID: 1 ---
Starting evaluation for ModelRun 1 (FlowerClassifier-Full-Test-734 1.0)...
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 514ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
....
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
Evaluation for ModelRun 1 completed.

🎉 Evaluation complete for ModelRun ID: 1
   - Total Cases: 734
   - Completed: 734
   - Failed: 0
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ python scripts/generate_report.py  1
--- Generating Report for ModelRun ID: 1 ---

==================================================
  Performance Report for: FlowerClassifier-Full-Test-734 (v1.0)
  Run ID: 1
==================================================

  Overall Accuracy: 79.43% (583/734 correct)

--- Category Performance ---
  - Daisy          : 80.00% (96/120)
  - Dandelion      : 80.50% (128/159)
  - Roses          : 79.39% (104/131)
  - Sunflowers     : 81.16% (112/138)
  - Tulips         : 76.88% (143/186)

--- Analysis of Failures ---
  1. Model failed on image: /home/dell-linux-dev3/Projects/ml-evaluation-framework/data/seeded_test_images/tulips/tulips_test_2.jpg
     - Ground Truth: 'tulips'
     - Prediction:   'roses'

  2. Model failed on image: /home/dell-linux-dev3/Projects/ml-evaluation-framework/data/seeded_test_images/daisy/daisy_test_2.jpg
     - Ground Truth: 'daisy'
     - Prediction:   'dandelion'

  3. Model failed on image: /home/dell-linux-dev3/Projects/ml-evaluation-framework/data/seeded_test_images/daisy/daisy_test_3.jpg
     - Ground Truth: 'daisy'
     - Prediction:   'tulips'

  4. Model failed on image: /home/dell-linux-dev3/Projects/ml-evaluation-framework/data/seeded_test_images/sunflowers/sunflowers_test_3.jpg
     - Ground Truth: 'sunflowers'
     - Prediction:   'tulips'

  5. Model failed on image: /home/dell-linux-dev3/Projects/ml-evaluation-framework/data/seeded_test_images/tulips/tulips_test_5.jpg
     - Ground Truth: 'tulips'
     - Prediction:   'dandelion'

  6. Model failed on image: /home/dell-linux-dev3/Projects/ml-evaluation-framework/data/seeded_test_images/tulips/tulips_test_7.jpg
     - Ground Truth: 'tulips'
     - Prediction:   'daisy'
....
  149. Model failed on image: /home/dell-linux-dev3/Projects/ml-evaluation-framework/data/seeded_test_images/tulips/tulips_test_185.jpg
     - Ground Truth: 'tulips'
     - Prediction:   'roses'

  150. Model failed on image: /home/dell-linux-dev3/Projects/ml-evaluation-framework/data/seeded_test_images/dandelion/dandelion_test_155.jpg
     - Ground Truth: 'dandelion'
     - Prediction:   'sunflowers'

  151. Model failed on image: /home/dell-linux-dev3/Projects/ml-evaluation-framework/data/seeded_test_images/sunflowers/sunflowers_test_135.jpg
     - Ground Truth: 'sunflowers'
     - Prediction:   'daisy'

/home/dell-linux-dev3/Projects/ml-evaluation-framework/scripts/generate_report.py:34: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
  ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha="right")

✅ Chart saved to: reports/run_1_accuracy_report.png

==================================================
  Report Complete
==================================================
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
Evaluation for ModelRun 1 completed.

🎉 Evaluation complete for ModelRun ID: 1
   - Total Cases: 734
   - Completed: 734
   - Failed: 0
(ml-eval-framework) dell-linux-d

if you start all anew yu gottta run the setup_db.py! 