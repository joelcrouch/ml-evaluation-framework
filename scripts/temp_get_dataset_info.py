import tensorflow_datasets as tfds
import sys
import os

# Add project root to path to allow imports, mirroring other scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    info = tfds.builder('tf_flowers').info
    print(info.splits['train'].num_examples)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)