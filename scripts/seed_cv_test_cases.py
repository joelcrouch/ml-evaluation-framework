import os
import requests
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from collections import defaultdict

# --- Configuration ---
API_URL = "http://localhost:8000/api/v1/prompts/"
IMAGE_OUTPUT_DIR = "data/seeded_test_images" # Use a new directory for the test set

def main():
    """
    Loads the TEST SPLIT of the 'tf_flowers' TFRecord dataset, saves images as JPGs,
    and seeds the database with TestCases pointing to these new images.
    """
    print(f"--- Seeding FULL CV Test Suite by extracting from TFRecords ---")

    # 1. Load the TEST dataset split from TFDS
    print("Loading 'tf_flowers' test split (train[80%:])...")
    test_ds, info = tfds.load('tf_flowers', split='train[80%:]', with_info=True)
    class_names = info.features['label'].names
    print(f"Dataset loaded. Found classes: {class_names}")

    # 2. Save images and create TestCases
    total_created = 0
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    
    # Keep track of image counts per category for unique naming
    image_counters = defaultdict(int)

    print("Processing test set and seeding database...")
    for example in test_ds.as_numpy_iterator():
        label_name = class_names[example['label']]
        image_tensor = example['image']
        
        # Create category subdirectory if it doesn't exist
        category_dir = os.path.join(IMAGE_OUTPUT_DIR, label_name)
        os.makedirs(category_dir, exist_ok=True)
        
        # Create a unique name for the image file
        image_counters[label_name] += 1
        image_name = f"{label_name}_test_{image_counters[label_name]}.jpg"
        image_path = os.path.join(category_dir, image_name)
        
        # Save the image tensor as a JPG file
        tf.keras.utils.save_img(image_path, image_tensor)
        
        # Use the absolute path for the TestCase
        absolute_image_path = os.path.abspath(image_path)
        
        payload = {
            "test_case_name": f"{label_name.capitalize()} Test {image_counters[label_name]}",
            "model_type": "image_classification",
            "input_type": "image_path",
            "output_type": "label",
            "input_data": {"image_path": absolute_image_path},
            "ground_truth": {"label": label_name}
        }

        try:
            response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            if response.status_code == 200:
                total_created += 1
                # Print progress intermittently
                if total_created % 50 == 0:
                    print(f"  ...created {total_created} TestCases so far...")
            else:
                print(f"  ❌ Failed to create TestCase for {absolute_image_path}. Status: {response.status_code}, Response: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Error: Could not connect to the API at {API_URL}.")
            print("Please ensure your FastAPI application (uvicorn) is running.")
            return

    print(f"\n🎉 Seeding complete. Created {total_created} new TestCases in '{IMAGE_OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
