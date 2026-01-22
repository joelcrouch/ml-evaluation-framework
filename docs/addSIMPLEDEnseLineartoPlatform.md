#  adding a dense model to the platform

This will be jsut like adding a simple model.  The differences between the simple linear and the dense layer lies in the method of training;

simople linear:
```
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
```
Dense:
```
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
```
It doesnt add 64 layers, it adds two hidden layer that have 64 units(neurons).  
1. Input: 19 features
2. Hidden layer 1: Dense layer with 64 units and a relu activation
3. Hidden layer 2: dense layer with 64 units and a relu acitivaton.
4. Output layer: Final dense layer with 1 unit to => single temp predction

This moves the model from a linear model to a neural network.  But waht do thay actually do?

1. Adding Hidden Layers (`Dense(units=64)`): The linear model could only learn a single weight for each input feature. By adding these "wider" hidden
      layers, we give the model more "thinking room." It now has thousands of parameters, allowing it to learn much more complex and subtle patterns from
      the input data.

2. Adding the Activation Function (`activation='relu'`): This is the most critical part. Without an activation function, stacking dense layers is
      mathematically pointless—a stack of linear operations is just another single linear operation.

      The relu (Rectified Linear Unit) activation function introduces non-linearity. It acts like a simple switch on each of the 64 units in a layer:
       * If the unit's output is positive, relu lets it pass through.
       * If the unit's output is negative, relu changes it to zero, effectively "turning it off."

  By combining layers of these simple on/off switches, the network can learn to approximate incredibly much more complex, non-linear relationships. It can learn
  "if-then" style rules like, "if the pressure is dropping and the wind is high, then the temperature will likely fall, but only if it's nighttime."  Stuff like that.  A
  simple linear model could never capture that kind of nuanced logic.

  In summary:

   * The Linear Model can only learn a single, straight-line relationship between the inputs and the output.
   * The Dense Model, by adding hidden layers and a relu activation, can learn complex, curved, and conditional relationships, giving it the potential to
     be a much more powerful and accurate model.

## quick explanation of activation
You might be wondering what activaton refers to, and what other options there are.  Good question. 
Here are some of the most common activation functions and when you might use them instead of relu:

  1. Sigmoid (or Logistic)

   * What it does: It's an "S"-shaped curve that squashes any input value into a range between 0 and 1.
   * When to use it: Almost exclusively for the output layer of a binary classification model. The output can be interpreted as the probability of the
     positive class. For example, in a "cat vs. dog" classifier, a sigmoid output of 0.9 would mean "90% probability that this is a dog." It's generally
     avoided in hidden layers today because it can lead to a problem called the "vanishing gradient," which makes deep networks hard to train.

  2. Tanh (Hyperbolic Tangent)

   * What it does: It's another "S"-shaped curve, very similar to sigmoid, but it squashes values into a range between -1 and 1.
   * When to use it: It used to be a popular choice for hidden layers before relu became dominant. Because its output is centered on zero, it can sometimes
     lead to faster convergence than sigmoid. It's still commonly found in the internal gates of some Recurrent Neural Network (RNN) architectures like
     LSTMs.

  3. Leaky ReLU

   * What it does: It's a small modification of relu. Instead of outputting 0 for all negative values, it allows a small, non-zero, negative slope (like y
     = 0.01x).
   * When to use it: This is a direct attempt to fix a potential problem with relu called the "Dying ReLU" problem, where a neuron can get "stuck" in the
     zero-output state and stop learning. Leaky ReLU ensures the neuron always has some gradient, allowing it to recover. It's often used as a drop-in
     replacement for relu and is considered by many to be a slight improvement.

  4. Softmax

   * What it does: It takes a vector of numbers and turns it into a probability distribution. All the output values are between 0 and 1, and they all sum
     up to 1.
   * When to use it: Exclusively for the output layer of a multi-class classification model. If you are classifying an image into one of 10 categories, the
     softmax layer will give you a list of 10 probabilities, one for each category, representing the model's confidence distribution across all classes.

  ---

  Why is relu the Default?

  ReLU became the go-to choice for hidden layers for two main reasons:
   1. It's Fast: The function max(0, x) is computationally very simple and cheap.
   2. It Fights Vanishing Gradients: Unlike sigmoid and tanh, its derivative is a constant 1 for all positive inputs. This allows the error signal
      (gradient) to flow back through deep networks much more effectively during training, which was a major breakthrough enabling the deep learning
      revolution.

  So, while you can certainly experiment with other activations like Leaky ReLU in your Dense model's hidden layers, relu remains a strong and reliable
  starting point.  Its easy and you can add it and it fits what we are trying to do.


But what is it?  It was inspired by the activation threshold required by a neuron before the neuron propagates a nerual spike (action potential-complex with lots of different ion based gates, leak gates all types of stuff)
he term "activation function" (or method) and its role in a neural network are directly inspired by the
  "activation threshold" of biological neurons in neuroscience. It's a wonderful example of biology inspiring computational models, and i like to think about hippocampus, and LTP generation, so its right in my wheelhouse.

  The Biological Analogy

  In your brain, a neuron works roughly like this:
   1. It receives electrochemical signals from many other neurons through its dendrites.
   2. It accumulates these incoming signals.
   3. If the total, accumulated signal is strong enough to cross a certain activation threshold, the neuron "fires," sending a sharp electrical spike (an
      "action potential") down its axon to pass the message along to other neurons.
   4. If the signal doesn't reach the threshold, the neuron stays silent and does nothing.

  The key idea is that a neuron is not just a passive wire; it's a decision-maker. It decides whether the incoming information is important enough to react
  to and pass on.

  The Artificial Neuron's Activation Function

  An artificial neuron (a node in a dense layer) mimics this process mathematically:

   1. Accumulate Signals: First, the neuron takes all its inputs, multiplies each one by a "weight" (representing the connection's strength), and sums them
      up. This gives a single number, the "weighted sum."

   2. Decide to "Fire": This is where the activation function comes in. The activation function takes the weighted sum as its input and determines the
      neuron's final output signal. It's the mathematical rule that decides how to "activate" based on the accumulated signal.

  Without an activation function, the neuron would just output the raw weighted sum, which is a simple linear operation. The activation function introduces
  the crucial element of non-linearity—the "firing" decision.

  Let's look at a few activation functions through this lens:

   * `ReLU` (`max(0, x)`): This is a very simple "firing" model.
       * If the weighted sum is negative (below the threshold of 0), the neuron outputs 0 ("stays silent").
       * If the weighted sum is positive (above the threshold), the neuron "fires," and the strength of its output signal is directly proportional to the
         strength of the input signal.

   * `Sigmoid`: This function models the probability of firing.
       * A large negative weighted sum results in an output near 0 (very low probability of firing).
       * A large positive weighted sum results in an output near 1 (very high probability of firing).

  In essence, the activation function is the mathematical heart of the artificial neuron, defining its behavior and allowing the network as a whole to
  learn complex patterns by controlling which neurons fire, when, and how strongly.


Ok.back to adding a dense model.

Here is the three-stage protocol for adding a new model to our platform:

   1. Stage 1: Train & Standardize: We'll create train_dense_time_series.py. This script will train the Dense model, save the final dense_model.keras file, and generate the dense_golden_dataset.json file.

   2. Stage 2: Seed the Database: We'll create seed_dense_test_cases.py to read the new golden dataset and populate our database with test cases, tagging them with the new time_series_dense model type.

   3. Stage 3: Integrate & Evaluate: We'll add a new elif block to scripts/run_evaluation.py to make the evaluation engine recognize the      "time_series_dense" type and connect it to the correct components.


###  Train and standardize
This is the only real part we have to change. I will outline the dfferences.


### Seed the database
This script is just like  the linear script.  We just change, essentailly, the metadata, and the assocated calls to (right now) curl to the api.
The only functional differences between the seed_linear_test_cases.py and seed_dense_test_cases.py scripts are:

   1. `GOLDEN_DATASET_PATH`: This changes to point to the correct golden dataset JSON file for the specific model (e.g., data/dense_golden_dataset.json).
   2. `payload["model_type"]`: This is updated to the unique identifier for the model (e.g., "time_series_dense").
   3. `payload["test_case_name"]`: Updated to reflect the model being seeded (e.g., "Dense TS Prediction - Case {case_id}").
   4. `payload["tags"]`: Updated to include relevant tags for the specific model (e.g., ["dense", "relu", ...]).
   5. Example `curl` command: Updated to match the new model_name and model_type for the next steps.

  So, while one change is a file path, all these modifications are indeed related to correctly identifying and categorizing the test cases in the database
  according to the specific model (linear vs. dense) they belong to. The core logic of making the API request remains unchanged.  While that looks like alot, its jsut basically metadata cahnges, and gives me an idea about how to abstract out this, and make this truly 'fire -and -forget'.

  ### quick idea save
 Let me describe a much more powerful and scalable, metadata-driven workflow.

  Is this professional MLOps best practice? I am proposing to separate the configuration from the code.

   * The Code: The generic scripts that know how to train, seed, and evaluate.
   * The Configuration: A central file that declares what to train and evaluate.

  Here’s exactly how that would look, building on this idea:

  Step 1: Create a Central Experiment File

  Instead of writing new scripts for each model, you would define all your experiments in a single YAML file, let's call it experiments.yaml. This file
  becomes the single source of truth for all your evaluation campaigns.

  `experiments.yaml`:

     experiments:
       - experiment_id: "linear_v1"
         model_name: "linear_model"
         model_type: "time_series_linear"
         train_script: "scripts/train_linear_time_series.py"
         params: {} # No special params for linear
     
       - experiment_id: "dense_2x64"
         model_name: "dense_2x64_model"
        model_type: "time_series_dense"
        train_script: "scripts/train_configurable_dense.py"
        params:
          layers: 2
          units: 64
          activation: "relu"
    
      - experiment_id: "dense_4_layers"
        model_name: "dense_4x64_model"
        model_type: "time_series_dense"
        train_script: "scripts/train_configurable_dense.py"
        params:
          layers: 4
          units: 64
          activation: "relu"

  Step 2: Use a Generic "Campaign Runner"

  Now, you'd have a single, powerful script (e.g., run_campaign.py) that does the following:

   1. It reads and parses experiments.yaml.
   2. It loops through each experiment defined in the file.
   3. For each experiment, it:
      
      a.  Calls the specified train_script, passing the params dictionary as command-line arguments.
      
      b.  The training script saves the model (dense_4x64_model.keras) and its golden dataset.
      
      c.  The campaign runner then makes the API call to create the ModelRun, saving the params dictionary into the config field of the database record.
      
      d.  Finally, it calls run_evaluation.py with the new run_id.
   4. This way you could loop thru a bunch of differnet architecture changes and determine which one is the 'best'.
    
  Step 3: Abstract the Seeding and Training

  This new workflow means you no longer need lots of separate scripts:

   * No More `seed_linear...`, `seed_dense...`: The seeding logic can be folded into a single, generic script or even handled directly by the campaign runner.
   * One Configurable Trainer: Instead of train_linear... and train_dense..., you could have one train_configurable_model.py that builds the model based on the params it receives.

  The Result

  With this system:

   * To test a new Dense model with 4 hidden layers, you don't write any new Python code. You simply add a new entry to your experiments.yaml file and run the campaign script.
   * Your database now stores not just the model's name but also the exact configuration of parameters used for that run, linked directly to the
     performance metrics.
   * Your report filenames, using our new convention, would automatically pick up this config (via the hash), giving you files like
     dense-4x64-model_v1.0_config-f0a1b2c3_run-101_performance-charts.png.

  This could be a path to evolving this from a series of individual scripts into a scalable, declarative, and far more powerful
  evaluation system.  And Uncle Clean code would be so proud! You might still have to write a custom seed script, and perhaps a custom reports generator. Maybe i can create the run_evaluation script with a bunch of flags for different architectures, or if that gets too cumbersome, creat different eval scripts for different tpes of models(becasue one eval does not fit or accurately/precisely evaluate a model).  Anyways, I digress. Lets not get into the weeds.


### Integrate and eval

For this "naive" Dense model (and the linear and baseline models from the tutorial), the WindowGenerator was configured with label_width=1.
  This means the model still only predicts one hour ahead.

  Because the output structure (a single-step prediction) is the same, our existing scripts/generate_report_time_series_v2.py script will work perfectly
  for the Dense model. It's designed to handle those single-step time series regression results and will correctly calculate MSE, MAE, and generate the
  plots.

  You only need a new report generation script (or significant modifications to the existing one) if you were to move to models that predict multiple hours
  ahead (i.e., label_width > 1). In that case, the structure of the prediction and ground_truth data would change from a single value to a sequence of
  values, and the plotting functions would need to adapt to visualize those sequences.  We will see that we need this requirement when we move onto a model that has a different shape, and output predictions are diffeent.


## brief expalansation of main chart
1. MSE Distribution (Mean Squared Error)

   * What It Is: This is a histogram showing the spread of the squared errors from all your test cases. The x-axis represents the MSE value, and the y-axis
     shows how many test cases resulted in that amount of error.
   * Why It Matters: Because the error is squared, this chart heavily emphasizes large mistakes. A prediction that is off by a little (e.g., 2 degrees) has
     a small squared error (4), but a prediction that is wildly off (e.g., 10 degrees) has a huge squared error (100).
   * How to Read It:
       * Ideal Shape: You want to see most of the bars clustered on the far left, very close to zero.
       * Long Tail: A long "tail" stretching to the right indicates that your model is making a few very large, significant errors on some test cases.
       * Mean vs. Median: The dotted lines are key. If the mean (average) is much higher than the median (middle value), it confirms that these large
         outlier errors are pulling up your average error score.

  2. MAE Distribution (Mean Absolute Error)

   * What It Is: This histogram shows the spread of the absolute errors (|true_value - predicted_value|).
   * Why It Matters: This is more intuitive than MSE because the error is in the same units as your prediction. If you are predicting temperature in
     Celsius, the MAE is also in Celsius. It answers the question, "Typically, how many degrees off was the model's prediction?"
   * How to Read It:
       * It gives you a more direct sense of the typical magnitude of the error across all test cases.
       * Unlike the MSE chart, it doesn't disproportionately penalize outliers, so it provides a better view of the model's "average" performance without
         being skewed by a few bad predictions.

  3. Test Set Score Distribution

   * What It Is: This histogram flips the perspective from error to "correctness." It shows the distribution of the normalized score for each test case,
     where 1.0 represents a perfect prediction (zero error) and 0.0 represents a very poor prediction.
   * Why It Matters: It's an easy-to-understand summary of how well the model performed.
   * How to Read It:
       * Ideal Shape: You want to see the vast majority of the bars pushed up against the right side, at or very near 1.0. This would mean your model is
         performing excellently on most test cases.
       * Left Tail: A tail stretching to the left indicates the proportion of test cases where the model struggled or failed.

  4. Test Set Error Metrics Box Plot

   * What It Is: This chart is a powerful way to summarize and compare the distributions of MSE, MAE, and RMSE all at once.
   * Why It Matters: It gives you a compact, statistical summary of your model's errors at a single glance.
   * How to Read It: For each metric (each box):
       * The Box itself represents the middle 50% of your data's errors.
       * The Line Inside the Box is the median error—the true middle value.
       * The Whiskers (the lines extending out) show the range of the vast majority of the errors.
       * Any Dots outside the whiskers are outliers—individual test cases where the error was unusually high compared to the rest. This instantly draws
         your attention to problematic predictions.


The other chart:prediction-samples.png chart shows a few randomly selected individual test cases (by default, it shows 5, but it's configurable with the --samples
  argument when you run the reporting script).

  For each of those selected samples, it plots the model's prediction (red 'X') against the actual ground truth (green circle) for the same point in time.
  It's a quick way to visually inspect the model's behavior on a handful of examples.


Pretty straightforward.