import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Dict, Any

class ImageClassifierModel:
    """
    A real image classification model that loads a pre-trained Keras model
    to classify images of flowers.
    """

    def __init__(self):
        """
        Initializes the model by loading the trained Keras model and
        the class names for the tf_flowers dataset.
        """
        print("--- Loading trained flower classifier model... ---")
        self.model = tf.keras.models.load_model('models/cv_flower_classifier.keras')
        
        # Get the class names from the dataset info
        _, info = tfds.load('tf_flowers', with_info=True, split='train[:1%]') # Load a small split to get info
        self.class_names = info.features['label'].names
        
        print(f"✅ Model loaded. Class names: {self.class_names}")

    def _preprocess_image(self, image_path: str):
        """Loads and preprocesses a single image."""
        img = tf.keras.utils.load_img(image_path, target_size=(160, 160))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        # The training script used 'tf.cast(image, tf.float32) / 255.0'
        # MobileNetV2 also has a dedicated preprocess_input function.
        # For consistency, we'll use the same division method.
        preprocessed_img = img_array / 255.0
        return preprocessed_img

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the class of a flower from an image file path.
        Expects 'image_path' in input_data.
        """
        image_path = input_data.get("image_path")
        if not image_path:
            raise ValueError("'image_path' not found in input_data")

        # Preprocess the image
        preprocessed_image = self._preprocess_image(image_path)

        # Get model prediction
        predictions = self.model.predict(preprocessed_image)
        
        # The output of the model is logits, we can apply softmax to get probabilities
        # but for finding the class, argmax is sufficient.
        predicted_index = np.argmax(predictions[0])
        predicted_label = self.class_names[predicted_index]

        return {"predicted_label": predicted_label}
