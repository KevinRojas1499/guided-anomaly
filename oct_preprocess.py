import os
import shutil

def copy_all_files_to_new_folder(source_dir, target_dir_name):
    # Create the target directory if it doesn’t exist
    target_dir = os.path.join(target_dir_name)
    print(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    # Walk through all subdirectories of the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Build the path to the source file
            file_path = os.path.join(root, file)
            # Replace hyphens with underscores in the file name
            new_file_name = file.replace('-', '_')
            # Build the path to where the file should be in the target directory
            target_file_path = os.path.join(target_dir, new_file_name)
            # Check to ensure the file is not being copied into the same directory
            if root != target_dir:
                # Copy file to the target directory
                shutil.copy2(file_path, target_file_path)
                #print(f”Copied {file} to {target_dir_name} folder”)
# Specify the directory where subfolders are located
source_directory = 'OCT_DATA/OCT2017/val' # Replace with your source directory path
target_directory = 'OCT_DATA/processed/val/'
# Run the function
copy_all_files_to_new_folder(source_directory,target_directory)