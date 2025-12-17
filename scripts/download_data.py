import tensorflow_datasets as tfds
import os

def main():
    """
    Downloads the 'tf_flowers' dataset using TensorFlow Datasets.
    """
    print("--- Downloading 'tf_flowers' dataset ---")
    print("This may take a few minutes...")

    # The data is typically downloaded to ~/tensorflow_datasets/ 
    # We just need to load it once to trigger the download.
    builder = tfds.builder("tf_flowers")
    builder.download_and_prepare()

    download_dir = os.path.join(os.path.expanduser("~"), "tensorflow_datasets")

    print("\n✅ 'tf_flowers' dataset downloaded and prepared successfully.")
    print(f"   Data is located in: {download_dir}")
    print("   You do not need to run this script again.")

if __name__ == "__main__":
    main()

