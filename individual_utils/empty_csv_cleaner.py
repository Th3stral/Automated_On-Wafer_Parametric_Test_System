import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pandas as pd

def remove_empty_csvs(folder_path):
    """
    Traverse all CSV files in a folder, try reading them with pandas,
    and delete the file if it is empty or cannot be read.

    :param folder_path: The path to the folder containing the CSV files.
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # List all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Process only CSV files
        if filename.endswith('.csv'):
            try:
                # Attempt to read the CSV file
                df = pd.read_csv(file_path)

                # Check if the DataFrame is empty
                if df.empty:
                    print(f"Deleting empty CSV file: {filename}")
                    os.remove(file_path)

            except Exception as e:
                # Handle cases where the file cannot be read
                print(f"Error reading file {filename}: {e}")
                print(f"Deleting unreadable CSV file: {filename}")
                os.remove(file_path)

if __name__ == "__main__":
    folder_path = input("Enter the folder path to scan for CSV files: ")
    remove_empty_csvs(folder_path)