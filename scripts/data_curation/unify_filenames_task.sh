#!/bin/bash

# Bash script directory
script_dir=$(dirname "$(readlink -f "$0")")

# Path to Python script
python_script_path="$script_dir/adapt_filename_in_tsv.py"

# Go to the data set directory
data_path="/scratch/data/2024FreeViewingMSCOCO"
cd "$data_path"

# Set target task
target_task="freeviewing"

# Find all files (and symbolic links) that have "task-" in their name (excluding xdf files)
find . \( -type f -o -type l \) -name "*task-*" -not -name "*.xdf"| while read -r file; do
    dir=$(dirname "$file")
    base=$(basename "$file")
    
    # Extract the task from the file name
    if [[ "$base" =~ (.*task-)([^_]+)(.*) ]]; then
        first_part="${BASH_REMATCH[1]}"
        task="${BASH_REMATCH[2]}"
        last_part="${BASH_REMATCH[3]}"

        # If the extracted task is different from the target task rename it accordingly
        if [[ "$task" != "$target_task" ]]; then
            new_base="${first_part}${target_task}${last_part}"
            new_file="$dir/$new_base"
            echo "Renaming $file to $new_file"
            mv "$file" "$new_file"
        fi

    fi
done

find . -type f -name "*sub-[0-9][0-9][0-9]_ses-[0-9][0-9][0-9]_scans.tsv" | while read -r file; do
    path=$(realpath "$file")
    echo "Processing: $path"
    python3 "$python_script_path" "$path" "$target_task"
done