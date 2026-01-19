#!/usr/bin/env python3

import pandas as pd
import sys

def update_filename_column(file_path, target_task):

    # Read the tsv file
    df = pd.read_csv(file_path, sep='\t')

    # Replace current task with target task if necessary
    if 'filename' in df.columns:
        new_filename = df['filename'].str.replace("task-([^_]+)", f"task-{target_task}", regex=True)

        if new_filename.ne(df['filename']).any():
            [print(f"Renaming {f_old} to {f_new}.") for f_old, f_new in zip(df['filename'], new_filename)]
            df['filename'] = new_filename

            # Save the changes
            df.to_csv(file_path, sep='\t', index=False)
            print(f"Updated {file_path}")

        else:
            [print(f"Filename {f_old} already matches the target {f_new} and does not need to be renamed.") for f_old, f_new in zip(df['filename'], new_filename)]

    else:
        print(f"'filename' column not found in {file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 adapt_filename_in_tsv.py <path_to_tsv_file> <target_task>")
        sys.exit(1)

    file_path = sys.argv[1]
    target_task = sys.argv[2]

    update_filename_column(file_path, target_task)