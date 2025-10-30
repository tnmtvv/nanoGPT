#!/bin/bash

mkdir -p csvs

# Use find to search recursively
find out-* -type f -name "*.csv" | while read csv_file; do
    # Get the parent directory name
    dir_name=$(dirname "$csv_file" | sed 's/.*\///')
    filename=$(basename "$csv_file")
    
    # Copy with directory prefix
    new_name="${dir_name}_${filename}"
    cp "$csv_file" "csvs/$new_name"
    echo "Copied: $csv_file -> csvs/$new_name"
done

echo "Done!"
