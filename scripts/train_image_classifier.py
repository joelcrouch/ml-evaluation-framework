import tensorflow as tf
import tensorflow_datasets as tfds
import os
#  this is not a complex model we use it b/c the data is freely available ant therer 
#are many tutorials about it
# Define constants
IMG_SIZE = 160 # All images will be resized to 160x160
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
EPOCHS = 5 # Keep epochs low for quick demonstration

def preprocess(image, label):
    """Resizes and normalizes images."""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0 # Normalize to [0,1]
    return image, label

def main():
    """Trains a simple image classifier on the tf_flowers dataset."""
    print("--- Starting Image Classifier Training ---")

    # 1. Load the tf_flowers dataset
    print("Loading tf_flowers dataset...")
    (train_ds, validation_ds), info = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:]'], # 80% train, 20% validation
        as_supervised=True, # Returns (image, label) tuples
        with_info=True
    )
    print("Dataset loaded.")

    num_classes = info.features['label'].num_classes
    print(f"Number of classes: {num_classes}")
    
    # Apply preprocessing and batching
    train_ds = train_ds.map(preprocess).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.map(preprocess).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Build the model (fine-tuning MobileNetV2)
    print("Building model (MobileNetV2 with fine-tuning)...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, # Don't include the classification head
        weights='imagenet' # Use pre-trained ImageNet weights
    )
    base_model.trainable = False # Freeze the pre-trained layers

    # Create a new classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes)

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    # 3. Compile the model
    print("Compiling model...")
    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    # 4. Train the model
    print(f"Training model for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=validation_ds
    )
    print("Training complete.")

    # 5. Save the trained model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "cv_flower_classifier.keras")
    model.save(model_path)
    print(f"✅ Trained model saved to: {model_path}")

if __name__ == "__main__":
    main()
