from typing import Dict, Any

class ImageClassifierModel:
    """
    A dummy image classification model for testing the framework.
    It simulates a model that classifies images.
    """

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates an image classification prediction.
        Expects 'image_path' in input_data.
        For demonstration, it always returns 'cat' as the prediction,
        unless 'dog' is in the image_path.
        """
        image_path = input_data.get("image_path")
        
        if image_path and "dog" in image_path.lower():
            predicted_label = "dog"
        else:
            predicted_label = "cat" # Default prediction

        return {"predicted_label": predicted_label}
