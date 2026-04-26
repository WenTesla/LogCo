#!/bin/bash
# set -euo pipefail

# Download the preprocessed data from Zenodo
# wget "https://zenodo.org/records/8115559/files/preprocessed.zip?download=1" -O preprocessed.zip

# Unzip the downloaded file
# unzip -o preprocessed.zip -d ./inputs

# Remove the downloaded zip file
# rm -f preprocessed.zip

# Keep only one structured file in each dataset folder
for dataset in HDFS BGL Spirit Thunderbird; do
  dataset_dir="./inputs/${dataset}"
  if [ ! -d "${dataset_dir}" ]; then
    echo "Skip: ${dataset_dir} not found"
    continue
  fi

  source_file="$(find "${dataset_dir}" -maxdepth 1 -type f \( -name "*.log_structured.csv" -o -name "*.log_struxx.csv" -o -name "*.log_stru*.csv" \) | head -n 1)"
  if [ -z "${source_file}" ]; then
    echo "Skip: no structured source file in ${dataset_dir}"
    continue
  fi

  mv -f "${source_file}" "${dataset_dir}/structured.csv"
  find "${dataset_dir}" -mindepth 1 ! -name "structured.csv" -delete
done

# Clean macOS unzip metadata if present
rm -rf ./inputs/__MACOSX
