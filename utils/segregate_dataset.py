import os
import shutil

# current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

dataset_folder = os.path.join(current_directory, "..", "dataset", "MRL_Eye_dataset")
root = os.path.join(current_directory, "..", "dataset")

# Create folders for "Closed_Eye" and "Opened_Eye"
closed_eye_folder = os.path.join(root, "Closed_Eyes")
opened_eye_folder = os.path.join(root, "Opened_Eyes")

os.makedirs(closed_eye_folder, exist_ok=True)
os.makedirs(opened_eye_folder, exist_ok=True)

# Iterate through each subfolder in the dataset
for subfolder in os.listdir(dataset_folder):
    subfolder_path = os.path.join(dataset_folder, subfolder)
    print("subfolder : ", subfolder)
    
    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Iterate through images in the subfolder and move them to the respective folders
        for image_file in os.listdir(subfolder_path):
            if image_file.endswith(".png"):
                image_path = os.path.join(subfolder_path, image_file)
                # Extract eye state from the image name
                eye_state = int(image_file.split("_")[4])                
                
                # Define the destination folder based on the eye state
                destination_folder = closed_eye_folder if eye_state == 0 else opened_eye_folder
                #print("image_file : ", image_file, "eye_state : ", eye_state, "destination_folder : ", destination_folder)

                # Move the image to the respective folder
                shutil.copy(image_path, os.path.join(destination_folder, image_file))
    print("done ", subfolder, "processing")
    
print("Images have been saved to 'Closed_Eye' and 'Opened_Eye' folders based on eye state.")
