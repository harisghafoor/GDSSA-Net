#!/bin/bash

# Create a directory to store the datasets
mkdir -p ../datasets

# Download/unzip TN3K dataset
tn3k_url="https://drive.google.com/uc?id=1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F"
tn3k_zip="../datasets/tn3k_dataset.zip"
tn3k_dir="../datasets/tn3k_dataset"
echo "Downloading TN3K dataset..."
curl -L -o "$tn3k_zip" "$tn3k_url" && unzip -q "$tn3k_zip" -d "$tn3k_dir" && rm "$tn3k_zip"

# # Download/unzip DDTI dataset (assuming it's another Google Drive link)
# ddti_url="YOUR_DDTI_https://drive.google.com/uc?id=1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50FDATASET_URL_HERE"
# ddti_zip="./datasets/ddti_dataset.zip"
# ddti_dir="./datasets/ddti_dataset"
# echo "Downloading DDTI dataset..."
# curl -L -o "$ddti_zip" "$ddti_url" && unzip -q "$ddti_zip" -d "$ddti_dir" && rm "$ddti_zip"

# echo "All datasets downloaded and unzipped."
