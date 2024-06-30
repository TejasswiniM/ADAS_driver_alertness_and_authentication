import os
import random
import shutil

# current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

def select_random_images(source_folder, destination_folder, num_images):
    # Get the list of all files in the source folder
    all_files = os.listdir(source_folder)

    # Randomly select num_images from the list
    selected_files = random.sample(all_files, num_images)

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Copy selected files to the destination folder
    for file_name in selected_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copyfile(source_path, destination_path)


# Replace 'path_to_source_folder' and 'path_to_destination_folder' with actual paths
#source_folder = os.path.join(current_directory, "..", "dataset", "Closed_Eyes")
#destination_folder = os.path.join(current_directory, "..", "dataset", "new_dataset_50_percent", "Closed_Eyes_20973")
#num_images_to_select = 20973

#source_folder = os.path.join(current_directory, "..", "dataset", "Opened_Eyes")
#destination_folder = os.path.join(current_directory, "..", "dataset", "new_dataset_50_percent", "Opened_Eyes_21476")
#num_images_to_select = 21476

#source_folder = os.path.join(current_directory, "..", "dataset", "Closed_Eyes")
#destination_folder = os.path.join(current_directory, "..", "dataset", "new_dataset_25_percent", "Closed_Eyes_10487")
#num_images_to_select = 10487

#source_folder = os.path.join(current_directory, "..", "dataset", "Opened_Eyes")
#destination_folder = os.path.join(current_directory, "..", "dataset", "new_dataset_25_percent", "Opened_Eyes_10738")
#num_images_to_select = 10738

#source_folder = os.path.join(current_directory, "..", "dataset", "Closed_Eyes")
#destination_folder = os.path.join(current_directory, "..", "dataset", "new_dataset_75_percent", "Closed_Eyes_31460")
#num_images_to_select = 31460

source_folder = os.path.join(current_directory, "..", "dataset", "Opened_Eyes")
destination_folder = os.path.join(current_directory, "..", "dataset", "new_dataset_75_percent", "Opened_Eyes_32214")
num_images_to_select = 32214

select_random_images(source_folder, destination_folder, num_images_to_select)
