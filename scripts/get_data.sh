#!/bin/bash

# Create a directory to store the datasets
mkdir -p ../datasets

# Download/unzip TN3K dataset
tn3k_url="https://drive.google.com/uc?id=1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F"
tn3k_zip="../datasets/tn3k_dataset.zip"
tn3k_dir="../datasets/tn3k_dataset"
echo "Downloading TN3K dataset..."
curl -L -o "$tn3k_zip" "$tn3k_url" && unzip -q "$tn3k_zip" -d "$tn3k_dir" && rm "$tn3k_zip"
