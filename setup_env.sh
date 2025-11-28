#!/bin/bash

# Creates and activates the Conda environment using environment.yml
#run conda activate ml-eval-framework after running the script
# --- Configuration ---
ENV_NAME="ml-eval-framework"
ENV_FILE="environment.yml"

echo "Starting environment setup..."

# 1. Check if the environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " RECREATE
    if [[ "$RECREATE" == "y" || "$RECREATE" == "Y" ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME
    else
        echo "Environment setup aborted."
        exit 0
    fi
fi

# 2. Create the environment from the YAML file
echo "Creating Conda environment '$ENV_NAME' from '$ENV_FILE'..."
conda env create -f $ENV_FILE

# 3. Handle potential errors during creation
if [ $? -eq 0 ]; then
    echo "✅ Conda environment '$ENV_NAME' created successfully."
    echo "To activate the environment, run:"
    echo "conda activate $ENV_NAME"
else
    echo "❌ ERROR: Failed to create the Conda environment."
fi